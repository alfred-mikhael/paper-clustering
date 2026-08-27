"""Upload embedded papers to Supabase"""

from paper_clustering.data_models import EmbeddedPaper
from supabase import Client


def upload(papers: list[EmbeddedPaper], client):
    pass
