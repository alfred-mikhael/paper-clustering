from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """Score proof-technique overlap independently of candidate retrieval."""

    def score_pairs(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 8
    ) -> list[float]:
        """Return one score per pair in input order; higher means more overlap.

        Scores are in [0, 1], but need not be calibrated probabilities.
        An empty input returns an empty list.
        """
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        *,
        top_k: int | None = None,
        batch_size: int = 8,
    ) -> list[tuple[int, float]]:
        """Return (candidate index, score) pairs in descending score order.

        Indices refer to the original candidates; ties preserve input order.
        ``top_k=None`` returns all candidates, and zero returns no results.
        """
        ...


class DatabaseClient(Protocol):
    def transaction(self) -> AbstractContextManager[Any]:
        """Commit grouped operations on success; roll back on an exception."""
        ...

    def insert(table: str, records: list[dict[str, Any]]) -> bool: ...

    def select(
        table: str, cols: list[str], where=list[tuple[str, str, Any]]
    ) -> list[list]: ...

    def execute_rpc(
        rpc_name: str, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]: ...
