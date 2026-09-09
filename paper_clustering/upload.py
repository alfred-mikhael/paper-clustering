"""Upload embedded papers to Supabase."""

from typing import Any, Protocol

from paper_clustering.data_models import ClusteredPapaer
from supabase import Client


class DatabaseClient(Protocol):
    def insert(records: list, db: str, table: str) -> Any: ...


def upload(papers: list[ClusteredPapaer], client: Client) -> Any | None:
    """Insert embedded papers into the ``papers`` Supabase table.

    Supabase's JSON encoder cannot serialize NumPy arrays, so vectors are
    converted to ordinary lists before insertion.  The table has three fixed
    technique columns; papers with fewer selected passages leave the remaining
    columns empty.
    """
    if not papers:
        return None

    records: list[dict[str, Any]] = []
    techniques: list[dict[str, Any]] = []
    for clustered_paper in papers:
        metadata = clustered_paper.embedded_paper.paper.metadata
        embeddings = clustered_paper.embedded_paper.embeddings
        x, y = clustered_paper.coords
        cluster = clustered_paper.cluster_label
        passages = list(embeddings.passages)
        vectors = list(embeddings.vectors)

        if len(passages) != len(vectors):
            raise ValueError(
                f"Paper {metadata.arxiv_id} has {len(passages)} passages but "
                f"{len(vectors)} technique vectors"
            )
        for vec, passage in zip(vectors, passages):
            techniques.append(
                {
                    "arxiv_id": metadata.arxiv_id,
                    "embedding": vec.to_list(),
                    "passage": passage,
                }
            )

        record: dict[str, Any] = {
            "arxiv_id": metadata.arxiv_id,
            "authors": ",".join(metadata.authors),
            "publication_date": metadata.publication_date.isoformat(),
            "title": metadata.title,
            "abstract": metadata.abstract,
            "url": metadata.url,
            "primary_category": metadata.primary_category,
            "area_embedding": embeddings.area_vector.tolist(),
            "umap_x": x,
            "umap_y": y,
            "cluster": cluster,
        }
        records.append(record)

    return (
        client.table("papers").insert(records).execute(),
        client.table("techniques").insert(techniques).execute(),
    )
