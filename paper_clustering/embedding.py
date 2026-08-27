"""Embed important passages into high-dimensional vectors."""

from sentence_transformers import SentenceTransformer
import numpy as np
from paper_clustering.data_models import TechniqueInfo, EmbeddingInfo


def embed_techniques(
    techniques: TechniqueInfo | list[TechniqueInfo], embedding_model
) -> EmbeddingInfo:
    pass


def embed(passages: list[str], embedding_model) -> list[np.array]:
    pass
