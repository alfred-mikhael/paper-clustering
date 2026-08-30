"""Embed important passages into high-dimensional vectors."""

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from paper_clustering.data_models import TechniqueInfo, EmbeddingInfo


def _choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_techniques(
    techniques: TechniqueInfo | list[TechniqueInfo],
    embedding_model: SentenceTransformer,
    *,
    batch_size: int = 8,
    prompt: str | None = None,
    device: str | None = None,
) -> EmbeddingInfo | list[EmbeddingInfo]:
    # split data into batches
    if techniques is not list:
        encoded = embedding_model.encode(
            techniques.passages,
            batch_size=batch_size,
            convert_to_numpy=True,
            prompt=prompt,
            device=_choose_device(device),
            normalize_embeddings=True,
        )
        return EmbeddingInfo(
            embedding_dim=embedding_model.get_embedding_dimension(),
            model_name=embedding_model.tokenizer.name_or_path,
            arxiv_id=techniques.arxiv_id,
            passages=techniques.passages,
            scores=techniques.scores,
            vectors=encoded,
        )
    else:
        results = []
        for tech in techniques:
            encoded = embedding_model.encode(
                tech.passages,
                batch_size=batch_size,
                convert_to_numpy=True,
                prompt=prompt,
                device=_choose_device(device),
                normalize_embeddings=True,
            )
            results.append(
                EmbeddingInfo(
                    embedding_dim=embedding_model.get_embedding_dimension(),
                    model_name=embedding_model.tokenizer.name_or_path,
                    arxiv_id=tech.arxiv_id,
                    passages=tech.passages,
                    scores=tech.scores,
                    vectors=encoded,
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
    )
