"""Extract important passages from a paper using a fine-tuned SciBERT classifier."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from paper_clustering.data_models import ArxivSection, Paper, Technique
from paper_clustering.embedding import embed, embed_techniques
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
    *,
    embedding_batch_size: int = 8,
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

    # Compare only passages from the same paper. Keep original indices so the
    # length-limited union order and final output order remain unchanged.
    indices_by_paper: dict[str, list[int]] = {}
    for index, technique in enumerate(techniques):
        indices_by_paper.setdefault(technique.arxiv_id, []).append(index)
    parents = list(
        (i, len(encoder.tokenizer.encode(t.text))) for i, t in enumerate(techniques)
    )

    def find(index: int) -> tuple[int, int]:
        root = index
        while parents[root][0] != root:
            root = parents[root][0]

        # Only the root's length is authoritative; compress the entire path.
        while index != root:
            parent = parents[index][0]
            parents[index] = parents[root]
            index = parent
        return parents[root]

    def union(left: int, right: int) -> None:
        left_root, left_length = find(left)
        right_root, right_length = find(right)
        if left_root == right_root:
            return

        merged_length = left_length + right_length + len("\n\n")
        if merged_length > 2048:
            return

        parents[left_root] = (left_root, merged_length)
        parents[right_root] = parents[left_root]

    for indices in indices_by_paper.values():
        if len(indices) < 2:
            continue
        paper_embeddings = embeddings[indices]
        # Embeddings are normalized, so dot products give cosine similarities.
        similarities = paper_embeddings @ paper_embeddings.T
        for row, left in enumerate(indices):
            for column in range(row + 1, len(indices)):
                right = indices[column]
                if (
                    similarities[row, column] >= threshold
                    and find(left)[1] + find(right)[1] <= 2048
                ):
                    union(left, right)

    groups: dict[tuple[int, int], list[Technique]] = {}
    for index, technique in enumerate(techniques):
        groups.setdefault(find(index), []).append(technique)

    merged_texts = [
        "\n\n".join(technique.text for technique in group)
        for group in groups.values()
        if len(group) > 1
    ]
    merged_embeddings = (
        iter(embed(merged_texts, encoder, batch_size=embedding_batch_size))
        if merged_texts else iter(())
    )

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
    pooling: str | None = None,
    technique_batch_size: int = 32,
    embedding_batch_size: int = 8,
) -> list[list[Technique]]:
    """Extract passages with optional mean/max pooling of classifier window scores.

    ``None`` truncates at 512 tokens. Pooling uses 512-token windows (including
    special tokens) with 128 passage tokens of overlap, preserving full text.
    """
    if pooling not in (None, "mean", "max"):
        raise ValueError('pooling must be None, "mean", or "max"')
    if technique_batch_size < 1 or embedding_batch_size < 1:
        raise ValueError("Technique and embedding batch sizes must be positive")
    datasets = [prepare_data(p.sections) for p in papers]
    # Score all paragraphs using the classifier's internal batching.
    paragraphs = [paragraph for _, _, _, text in datasets for paragraph in text]
    logger.info(
        "processed %d papers with a total of %d paragraphs",
        len(papers),
        len(paragraphs),
    )
    predictions = classifier.predict(
        paragraphs, pooling=pooling, batch_size=technique_batch_size
    )

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

    embeddings = embed(
        [text for _, _, _, text, _ in candidates], encoder,
        batch_size=embedding_batch_size,
    )
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
    techniques = merge_similar(
        techniques, encoder, threshold, embedding_batch_size=embedding_batch_size
    )

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
            if technique.score < min(k + index / 4.0, 3.9):
                break
            selected.append(technique)
        selected_by_paper.append(selected)

    return selected_by_paper


# Backwards-compatible spelling for the original draft name.
extract_technique_and_embed_batch = extract_techniques_and_embed_batch
