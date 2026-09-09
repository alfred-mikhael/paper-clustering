import numpy as np
from supabase import Client
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    query_arxiv_id: str
    target_arxiv_id: str
    query_passage: str
    target_passage: str
    similarity: float


def find_similar(
    arxiv_id: str, client: Client, k: int = 10
) -> dict[str, list[SearchResult]]:
    """Returns a list of the arxiv ids, technique ids, and similarity scores of the top k most similar papers by technique."""
    resp = (
        client.table("techniques")
        .select("embedding", "passage", "arxiv_id")
        .eq("arxiv_id", arxiv_id)
        .execute()
    )
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
    query: np.ndarray, client: Client, k: int = 10
) -> list[tuple[str, str, float]]:
    res = client.rpc(
        "ANN",
        {
            "query_embedding": query.to_list(),
            "match_count": k,
        },
    ).execute()
    return [(row[1], row[2], row[3]) for row in res.data]


def _aggregate_results(
    candidates: list[SearchResult], keys: set[str]
) -> dict[str, list[SearchResult]]:
    agg = dict()
    for k in keys:
        agg[k] = []
    for c in candidates:
        agg[c.target_arxiv_id].append(c)
    return agg
