from dataclasses import dataclass
from datetime import datetime
import numpy as np


# For later use
@dataclass(frozen=True)
class CitationRecord:
    authors: tuple[str, ...]
    doi: str | None
    arxiv_id: str | None
    title: str


@dataclass(frozen=True)
class PaperMetadata:
    authors: tuple[str, ...]
    publication_date: datetime
    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    url: str
    doi: str
    # cites: list[str]


@dataclass(frozen=True)
class ArxivSection:
    """A cleaned paper section returned by :func:`get_sections`."""

    arxiv_id: str
    section_index: int
    section_header: str
    major_section_header: str
    text: tuple[str, ...]


@dataclass(frozen=True)
class Technique:
    """An important technique in a paper"""

    arxiv_id: str
    text: str
    embedding: np.ndarray
    score: float


@dataclass(frozen=True)
class Paper:
    metadata: PaperMetadata
    sections: tuple[ArxivSection, ...]


@dataclass(frozen=True, kw_only=True)
class EmbeddedPaper(Paper):
    """A paper which has an embedded area vector and a list of embedded techniques"""

    embedding_dim: int
    model_name: str
    arxiv_id: str
    area_vector: np.ndarray
    techniques: list[Technique]
    coords: tuple[float, float]
    cluster_label: int


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
