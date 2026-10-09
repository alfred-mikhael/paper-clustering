import numpy as np
from dataclasses import dataclass
from paper_clustering.interfaces import DatabaseClient, Reranker


@dataclass(frozen=True)
class SearchResult:
    query_arxiv_id: str
    target_arxiv_id: str
    target_title: str
    query_passage: str
    target_passage: str
    similarity: float


def find_similar(
    arxiv_id: str,
    client: DatabaseClient,
    k: int = 10,
    reranker: Reranker | None = None,
) -> dict[str, list[SearchResult]]:
    """Return similar papers grouped by ID, optionally ordered by reranker score.

    Retrieve up to k candidates per query passage. When a reranker is supplied,
    rank papers by their best passage-pair score and order each paper's pairs by
    that score. SearchResult.similarity remains the original vector similarity.
    Without a reranker, use the existing vector-similarity ranking.
    """
    candidates = retrieve_candidates(arxiv_id, client, k=k)
    candidates = [c for c in candidates if c.target_arxiv_id != c.query_arxiv_id]
    return rank_candidates(candidates, k=k, reranker=reranker)


def retrieve_candidates(
    arxiv_id: str, client: DatabaseClient, k: int = 10
) -> list[SearchResult]:
    """Materialize all candidate pairs before releasing the database connection."""
    resp = client.select(
        "techniques",
        cols=["embedding", "passage", "arxiv_id"],
        where=[("arxiv_id", "eq", arxiv_id)],
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
                target_title=title,
            )
            for target_id, passage, sim, title in _find_similar_vectors(
                row["embedding"], client, k=k + 1
            )
            if arxiv_id != target_id
        ]
    return candidates


def rank_candidates(
    candidates: list[SearchResult],
    k: int = 10,
    reranker: Reranker | None = None,
) -> dict[str, list[SearchResult]]:
    """Rank and group retrieved pairs without accessing the database."""
    if reranker is not None and candidates:
        scores = reranker.score_pairs(
            [(c.query_passage, c.target_passage) for c in candidates]
        )
        if len(scores) != len(candidates):
            raise ValueError("Reranker must return one score per passage pair")
        # Stable sorting preserves retrieval order for equal reranker scores.
        candidates = [
            candidate
            for candidate, score in sorted(
                zip(candidates, scores), key=lambda pair: pair[1], reverse=True
            )
        ]
        # The first occurrence of a paper is its highest-scoring passage pair.
        topk_ids = list(dict.fromkeys(c.target_arxiv_id for c in candidates))[:k]
    else:
        topk_ids = _find_topk(candidates, k=k)
    return _aggregate_results(
        [c for c in candidates if c.target_arxiv_id in topk_ids], topk_ids
    )


def _find_topk(candidates: list[SearchResult], k: int) -> list[str]:
    seen = list()
    for result in sorted(candidates, key=lambda res: res.similarity, reverse=True):
        if len(seen) >= k:
            return seen
        if result.target_arxiv_id not in seen:
            seen.append(result.target_arxiv_id)
    return seen


def _find_similar_vectors(
    query: np.ndarray, client: DatabaseClient, k: int
) -> list[tuple[str, str, float, str]]:
    """Returns a list of (arxiv_id, passage, similarity) of most similar passages, deduplicated by paper"""
    res = client.execute_rpc(
        "ann",
        {
            "query_vector": query,
            "match_count": k,
        },
    )
    return [
        (row["arxiv_id"], row["passage"], row["similarity"], row["title"])
        for row in res
    ]


def _aggregate_results(
    candidates: list[SearchResult], keys: list[str]
) -> dict[str, list[SearchResult]]:
    agg = dict()
    for k in keys:
        agg[k] = []
    for c in candidates:
        agg[c.target_arxiv_id].append(c)
    return agg
