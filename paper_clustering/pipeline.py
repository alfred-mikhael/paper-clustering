from paper_clustering.data_models import (
    Paper,
    PaperMetadata,
    EmbeddedPaper,
    EmbeddingInfo,
    ClusteredPapaer,
)
from supabase import Client
import numpy as np
from sentence_transformers import SentenceTransformer
from paper_clustering.technique_classifier import SciBERTClassifier
from paper_clustering.extract_text import get_paper
from paper_clustering.extract_techniques import extract_techniques_and_embed
from paper_clustering.cluster import generate_clusters
from paper_clustering.generate_coords import generate_umap
from paper_clustering.upload import upload
import requests
import os

ARXIV_BASE_URL = ...
ARXIV_REQUEST_WAIT_TIME = ...


def get_relevant(
    days_back: int,
    max_results: int = 10000,
    categories: list[str] | None = None,
    keywords: list[str] | None = None,
) -> list[str]:
    """Gets a list of arxiv_ids published at most `days_back` days ago which match
    the provided `categories` and `keywords`."""
    pass


def get_month_from_batch_api():
    """Download arxiv source files from the S3 API, and process only those papers which are relevant."""
    pass


def upload_all():
    client = Client(os.environ("SUPABASE_URL"), os.environ("SUPABASE_KEY"))
    classifier = SciBERTClassifier()
    encoder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
    )
    ...
