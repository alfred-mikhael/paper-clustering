"""Extract important passages from a paper using a fine-tuned SciBERT classifier."""

from paper_clustering.data_models import (
    Paper,
    TechniqueInfo,
    ArxivSection,
    EmbeddingInfo,
)
from paper_clustering.technique_classifier import TechniqueClassifier
from paper_clustering.embedding import embed
from sentence_transformers import SentenceTransformer
import numpy as np
from dppy.finite_dpps import FiniteDPP
import logging


def prepare_data(
    sections: list[ArxivSection],
) -> tuple(list[int], list[int], list[str]):
    section_indices: list[int] = []
    paragraph_indices: list[int] = []
    texts: list[str] = []
    for section in sections:
        header = section.section_header
        paragraph_index = 0
        for par in section.text:
            section_indices.append(section.section_index)
            paragraph_indices.append(paragraph_index)
            texts.append(f"Section: {section} Text: {par}")
            paragraph_index += 1
    return section_indices, paragraph_indices, texts


def merge_similar(
    passages: list[str], encoder: SentenceTransformer, threshold: float = 0.85
) -> list[str]:
    """Merge semantically similar passages.

    Does so by constructing a pairwise similarity matrix for the embeddings of the text, and then
    concatenating those `passages` whose embeddings have cosine similarity at least `threshold`.
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1")
    if not passages:
        return []

    embeddings = np.asarray(embed(passages, encoder), dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(passages):
        raise ValueError("encoder must return one embedding vector per passage")

    # ``embed`` requests normalized vectors, so their dot product is cosine
    # similarity. Connected components make merging independent of input order:
    # a passage joins a group when it is similar to any member of that group.
    similarities = embeddings @ embeddings.T
    parents = list(range(len(passages)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    row_indices, column_indices = np.triu_indices(len(passages), k=1)
    for left, right in zip(row_indices, column_indices, strict=True):
        if similarities[left, right] >= threshold:
            union(int(left), int(right))

    groups: dict[int, list[str]] = {}
    for index, passage in enumerate(passages):
        groups.setdefault(find(index), []).append(passage)
    return ["\n\n".join(group) for group in groups.values()]


def select_most_diverse(
    passages: list[str],
    encoder: SentenceTransformer,
    k: int,
    prompt: str = "clustering",
) -> list[tuple[str, np.ndarray]]:
    """Selects the k most diverse passages by sampling from a k-DPP.

    If k < len(passages), will return the len(passages) string-embedding pairs, and k - len(passages) empty strings
    """
    if k > len(passages):
        logging.warn(
            f"k={k} is larger than the total number of passages {len(passages)}."
        )
        return [
            (p, v) for p, v in zip(passages, embed(passages, encoder, prompt=prompt))
        ] + [
            (
                "",
                np.zeros(
                    encoder.get_embedding_dimension(),
                ),
            )
        ] * (
            k - len(passages)
        )
    embeddings = embed(passages, encoder, prompt=prompt)
    M = np.vstack(embeddings)
    DPP = FiniteDPP("likelihood", **{"L": M @ M.T})
    DPP.flush_samples()
    indices = DPP.sample_exact_k_dpp(size=k)
    return [(passages[i], embeddings[i]) for i in indices]


def extract_techniques(
    paper: Paper,
    classifier: TechniqueClassifier,
    encoder: SentenceTransformer,
    *,
    merge: bool = True,
    k: int = 3,
) -> EmbeddingInfo:
    sections = paper.sections
    section_indices, paragraph_indices, dataset = prepare_data(sections)
    keyed_predictions = list(
        zip(section_indices, paragraph_indices, classifier.predict(dataset))
    )
    keyed_predictions.sort(key=lambda key_pred: key_pred[2])
    top20 = [
        f"Section: {sections[i].section_header} Text: {sections[i].text[j]}"
        for i, j, _ in keyed_predictions[:20]
    ]
    candidates = merge_similar(top20, encoder) if merge else top20[:10]
    diverse = select_most_diverse(candidates, encoder, k)
    return EmbeddingInfo(
        embedding_dim=encoder.get_embedding_dimension(),
        model_name=encoder.tokenizer.name_or_path,
        arxiv_id=paper.metadata.arxiv_id,
        passages=[pair[0] for pair in diverse],
        vectors=[pair[1] for pair in diverse],
    )
