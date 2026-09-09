"""Extract important passages from a paper using a fine-tuned SciBERT classifier."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from paper_clustering.data_models import ArxivSection, Paper, Technique
from paper_clustering.embedding import embed
from paper_clustering.technique_classifier import TechniqueClassifier

logger = logging.getLogger(__name__)


def prepare_data(
    sections: list[ArxivSection],
) -> tuple[list[int], list[int], list[str], list[str]]:
    section_indices: list[int] = []
    paragraph_indices: list[int] = []
    ids: list[str] = []
    texts: list[str] = []
    for section in sections:
        header = section.section_header
        paragraph_index = 0
        for par in section.text:
            section_indices.append(section.section_index)
            paragraph_indices.append(paragraph_index)
            ids.append(section.arxiv_id)
            texts.append(f"Section: {header} Text: {par}")
            paragraph_index += 1
    return section_indices, paragraph_indices, ids, texts


def merge_similar(
    techniques: list[Technique],
    encoder: SentenceTransformer,
    threshold: float = 0.90,
) -> list[Technique]:
    """Merge semantically similar techniques from the same paper.

    Connected groups of techniques whose embeddings have cosine similarity at
    least ``threshold`` are concatenated. Techniques with different
    ``arxiv_id`` values are never merged. Each concatenated passage is embedded
    again so that its vector represents the complete merged text.
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1")
    if not techniques:
        return []

    embeddings = np.asarray([technique.embedding for technique in techniques])
    if embeddings.ndim != 2 or embeddings.shape[0] != len(techniques):
        raise ValueError("techniques must contain one embedding vector each")

    # ``embed`` requests normalized vectors, so their dot product is cosine
    # similarity. Connected components make merging independent of input order:
    # a passage joins a group when it is similar to any member of that group.
    similarities = embeddings @ embeddings.T
    parents = list(range(len(techniques)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    row_indices, column_indices = np.triu_indices(len(techniques), k=1)
    for left, right in zip(row_indices, column_indices, strict=True):
        if (
            techniques[left].arxiv_id == techniques[right].arxiv_id
            and similarities[left, right] >= threshold
        ):
            union(int(left), int(right))

    groups: dict[int, list[Technique]] = {}
    for index, technique in enumerate(techniques):
        groups.setdefault(find(index), []).append(technique)

    merged_texts = [
        "\n\n".join(technique.text for technique in group)
        for group in groups.values()
        if len(group) > 1
    ]
    merged_embeddings = iter(embed(merged_texts, encoder)) if merged_texts else iter(())

    merged: list[Technique] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        best = max(group, key=lambda technique: technique.score)
        merged.append(
            Technique(
                arxiv_id=best.arxiv_id,
                text="\n\n".join(technique.text for technique in group),
                embedding=next(merged_embeddings),
                score=best.score,
            )
        )

    logger.info("%d total merges", len(techniques) - len(merged))
    return merged


def extract_techniques_and_embed_batch(
    papers: list[Paper],
    classifier: TechniqueClassifier,
    encoder: SentenceTransformer,
    *,
    threshold: float = 0.9,
    k: int = 3,
) -> list[list[Technique]]:
    datasets = [prepare_data(p.sections) for p in papers]
    # Run the classifier on every paragraph in a single batch.
    paragraphs = [paragraph for _, _, _, text in datasets for paragraph in text]
    logger.info(
        "processed %d papers with a total of %d paragraphs",
        len(papers),
        len(paragraphs),
    )
    predictions = classifier.predict(paragraphs)

    # Get the top candidate paragraphs from each paper.
    offset = 0
    best_per_paper: list[list[tuple[int, int, str, str, float]]] = []
    for dataset in datasets:
        section_indices, paragraph_indices, ids, texts = dataset
        keyed_predictions = list(
            zip(
                section_indices,
                paragraph_indices,
                ids,
                texts,
                predictions[offset : offset + len(section_indices)],
                strict=True,
            )
        )
        offset += len(section_indices)
        keyed_predictions.sort(key=lambda prediction: prediction[4], reverse=True)
        best_per_paper.append(keyed_predictions[:20])

    candidates = [
        candidate
        for paper_candidates in best_per_paper
        for candidate in paper_candidates
    ]
    if not candidates:
        return [[] for _ in papers]

    embeddings = embed([text for _, _, _, text, _ in candidates], encoder)
    techniques = [
        Technique(
            arxiv_id=arxiv_id,
            text=text,
            embedding=embedding,
            score=float(score),
        )
        for (_, _, arxiv_id, text, score), embedding in zip(
            candidates, embeddings, strict=True
        )
    ]
    techniques = merge_similar(techniques, encoder, threshold)

    # Always keep the best technique from each paper. Keep the technique at
    # rank i only when its classifier score is at least k + i / 3.
    techniques_by_id: dict[str, list[Technique]] = {}
    for technique in techniques:
        techniques_by_id.setdefault(technique.arxiv_id, []).append(technique)

    selected_by_paper: list[list[Technique]] = []
    for paper in papers:
        ranked = sorted(
            techniques_by_id.get(paper.metadata.arxiv_id, []),
            key=lambda technique: technique.score,
            reverse=True,
        )
        selected = ranked[:1]
        for index, technique in enumerate(ranked[1:], start=1):
            if technique.score < k + index / 3.0:
                break
            selected.append(technique)
        selected_by_paper.append(selected)

    return selected_by_paper


# Backwards-compatible spelling for the original draft name.
extract_technique_and_embed_batch = extract_techniques_and_embed_batch
