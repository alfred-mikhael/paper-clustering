"""Upload embedded papers to database."""

from typing import Any

from paper_clustering.data_models import EmbeddedPaper
from paper_clustering.utils import DatabaseClient


def upload(
    papers: list[EmbeddedPaper], client: DatabaseClient
) -> tuple[bool, bool] | None:
    """Insert embedded papers into the ``papers`` database table.

    Supabase's JSON encoder cannot serialize NumPy arrays, so vectors are
    converted to ordinary lists before insertion.  The table has three fixed
    technique columns; papers with fewer selected passages leave the remaining
    columns empty.
    """
    if not papers:
        return None

    records: list[dict[str, Any]] = []
    techniques: list[dict[str, Any]] = []
    for paper in papers:
        metadata = paper.metadata
        x, y = paper.coords

        for technique in paper.techniques:
            techniques.append(
                {
                    "arxiv_id": metadata.arxiv_id,
                    "embedding": technique.embedding.tolist(),
                    "passage": technique.text,
                    "score": technique.score,
                }
            )

        record: dict[str, Any] = {
            "arxiv_id": metadata.arxiv_id,
            "authors": metadata.authors,
            "publication_date": metadata.publication_date.isoformat(),
            "title": metadata.title,
            "abstract": metadata.abstract,
            "url": metadata.url,
            "primary_category": metadata.primary_category,
            "area_embedding": paper.area_vector.tolist(),
            "umap_x": x,
            "umap_y": y,
        }
        records.append(record)

    return (
        client.insert(table="papers", records=records),
        client.insert(table="techniques", records=techniques),
    )
