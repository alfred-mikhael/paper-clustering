import numpy as np
from dataclasses import dataclass
from utils import DatabaseClient


@dataclass(frozen=True)
class SearchResult:
    query_arxiv_id: str
    target_arxiv_id: str
    query_passage: str
    target_passage: str
    similarity: float


def find_similar(
    arxiv_id: str, client: DatabaseClient, k: int = 10
) -> dict[str, list[SearchResult]]:
    """Returns a list of the arxiv ids, technique ids, and similarity scores of the top k most similar papers by technique."""
    resp = client.select(
        "techniques",
        cols=["embedding", "passage", "arxiv_id"],
        where=[("eq", "arxiv_id", arxiv_id)],
    )
    # supabase implementation
    # resp = (
    #     client.table("techniques")
    #     .select("embedding", "passage", "arxiv_id")
    #     .eq("arxiv_id", arxiv_id)
    #     .execute()
    # )
    candidates = []
    for row in resp:
        candidates += [
            SearchResult(
                query_arxiv_id=arxiv_id,
                target_arxiv_id=target_id,
                query_passage=row["passage"],
                target_passage=passage,
                similarity=sim,
            )
            for target_id, passage, sim in _find_similar_vectors(
                row["embedding"], client
            )
        ]
    # This does deduplication, but my current postgres implementation already does that
    # I am keeping it only because it has minimal performance impact and for safety.
    topk_ids = _find_topk(candidates)
    return _aggregate_results(
        [c for c in candidates if c.target_arxiv_id in topk_ids], topk_ids
    )


def _find_topk(candidates: list[SearchResult]) -> set[str]:
    seen = set()
    for result in sorted(candidates, key=lambda res: res.similarity, reverse=True):
        if len(seen) >= 10:
            return seen
        seen.add(result.target_arxiv_id)
    return seen


def _find_similar_vectors(
    query: np.ndarray, client: DatabaseClient, k: int = 10
) -> list[tuple[str, str, float]]:
    """Returns a list of (arxiv_id, passage, similarity) of most similar passages, deduplicated by paper"""
    res = client.execute_rpc(
        "ANN",
        {
            "query_embedding": query.to_list(),
            "match_count": k,
        },
    )
    return [(row["arxiv_id"], row["passage"], row["similarity"]) for row in res.data]


def _aggregate_results(
    candidates: list[SearchResult], keys: set[str]
) -> dict[str, list[SearchResult]]:
    agg = dict()
    for k in keys:
        agg[k] = []
    for c in candidates:
        agg[c.target_arxiv_id].append(c)
    return agg
