"""Embed important passages into high-dimensional vectors."""

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from paper_clustering.data_models import Technique


def _choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_techniques(
    techniques: str | list[str],
    scores: float | list[float],
    arxiv_ids: str | list[str],
    embedding_model: SentenceTransformer,
    *,
    batch_size: int = 8,
    prompt: str | None = None,
    device: str | None = None,
) -> Technique | list[Technique]:
    # split data into batches
    if not isinstance(techniques, list):
        encoded = embedding_model.encode(
            techniques,
            batch_size=batch_size,
            convert_to_numpy=True,
            prompt=prompt,
            device=_choose_device(device),
            normalize_embeddings=True,
        )
        return Technique(
            arxiv_id=arxiv_ids,
            text=techniques,
            score=scores,
            embedding=encoded,
        )
    else:
        results = []
        encoded = embedding_model.encode(
            techniques,
            batch_size=batch_size,
            convert_to_numpy=True,
            prompt=prompt,
            device=_choose_device(device),
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for i, e in enumerate(encoded):
            results.append(
                Technique(
                    arxiv_id=arxiv_ids[i],
                    text=techniques[i],
                    embedding=e,
                    score=scores[i],
                )
            )
        return results


def embed(
    passages: list[str],
    embedding_model: SentenceTransformer,
    *,
    batch_size: int = 8,
    prompt: str | None = None,
    device: str | None = None,
) -> list[np.ndarray]:
    return embedding_model.encode(
        passages,
        prompt=prompt,
        batch_size=batch_size,
        convert_to_numpy=True,
        device=_choose_device(device),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
