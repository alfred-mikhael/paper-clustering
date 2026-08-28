from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass(frozen=True)
class PaperMetadata:
    authors: tuple[str, ...]
    publication_date: datetime
    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    url: str


@dataclass(frozen=True)
class ArxivSection:
    """A cleaned paper section returned by :func:`get_sections`."""

    arxiv_id: str
    section_index: int
    section_header: str
    major_section_header: str
    text: tuple[str, ...]


@dataclass(frozen=True)
class TechniqueInfo:
    """Important text passages in a paper"""

    arxiv_id: str
    passages: tuple[str, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class EmbeddingInfo:
    """Important text passages in a paper and their embedding vectors"""

    embedding_dim: int
    model_name: str
    arxiv_id: str
    passages: tuple[str, ...]
    scores: tuple[float, ...]
    vectors: tuple[np.ndarray]


@dataclass(frozen=True)
class Paper:
    metadata: PaperMetadata
    sections: tuple[ArxivSection, ...]


@dataclass(frozen=True)
class EmbeddedPaper:
    paper: Paper
    embeddings: EmbeddingInfo


@dataclass(frozen=True)
class ModelMetrics:
    k: int
    eval_ids: list[str]
    mean_precision_at_k: float
    brier_score: float
    bce: float
    macro_f1: float
    classwise_accuracy: tuple[float, ...]
    classwise_recall: tuple[float, ...]
    classwise_precision: tuple[float, ...]
    classwise_f1: tuple[float, ...]
