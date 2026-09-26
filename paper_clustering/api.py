"""HTTP access to paper-technique search.

Run from the repository root:
    uvicorn paper_clustering.api:app --host 127.0.0.1 --port 8000

Set DATABASE_URL to a PostgreSQL connection string, or use the standard
PGHOST, PGPORT, PGDATABASE, PGUSER and PGPASSWORD environment variables.
Interactive API documentation is available at /docs.
Install psycopg[binary,pool] for connection pooling. Use one Uvicorn worker
on a single GPU: each worker loads its own reranker and connection pool.
"""

import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from threading import Lock
from typing import Annotated

import psycopg
from fastapi import FastAPI, HTTPException, Query, Request
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

from paper_clustering.postgres import PostgresClient
from paper_clustering.query import SearchResult, rank_candidates, retrieve_candidates
from paper_clustering.reranker import SLMReranker
from paper_clustering.utils import DatabaseClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources at startup and release them at shutdown."""
    pool = ConnectionPool(
        conninfo=os.environ.get("DATABASE_URL", ""),
        min_size=2,
        max_size=10,
        timeout=10,
        kwargs={"autocommit": True, "connect_timeout": 10},
        # Register pgvector once for each new physical connection, including
        # replacement connections created by the pool later.
        configure=register_vector,
        open=False,
    )
    try:
        # Fail startup if the database is unavailable, before loading the model.
        pool.open(wait=True, timeout=10)
        app.state.database_pool = pool
        app.state.reranker = SLMReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
        app.state.reranker_lock = Lock()
        yield
    finally:
        pool.close()
        # Drop the application's model reference on shutdown.
        if hasattr(app.state, "reranker"):
            del app.state.reranker


app = FastAPI(title="Paper Technique Search", lifespan=lifespan)


@contextmanager
def get_database_client(request: Request) -> Iterator[DatabaseClient]:
    """Lend a pooled connection only for the caller's database operations."""
    try:
        # The client is lightweight and request-local; its physical connection
        # is reused. The context returns it even if retrieval raises an error.
        with request.app.state.database_pool.connection() as connection:
            yield PostgresClient(connection)
    except (
        psycopg.OperationalError,
        psycopg.InterfaceError,
        PoolTimeout,
        PoolClosed,
    ) as error:
        # Keep connection details in server logs, not in the HTTP response.
        logger.exception("Database unavailable during paper search")
        raise HTTPException(status_code=503, detail="Database unavailable") from error


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/query", response_model=dict[str, list[SearchResult]])
def query_papers(
    request: Request,
    arxiv_id: Annotated[str, Query(min_length=1, pattern=r"\S")],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> dict[str, list[SearchResult]]:
    """Find up to k similar papers, grouped by target arXiv ID.

    Papers are ordered by their best reranker score; each match's similarity
    remains the original vector similarity. An empty object
    means no matches were found, including when the query has no stored passages.
    """
    # FastAPI returns HTTP 422 for invalid query parameters.
    # A synchronous endpoint runs in a worker thread so blocking database calls
    # do not block the async event loop. Dataclasses are serialized to JSON using
    # the response_model above, which also documents the result fields in /docs.
    with get_database_client(request) as client:
        candidates = retrieve_candidates(arxiv_id.strip(), client, k=k)
    # The connection is back in the pool before waiting for the GPU. Preserve
    # the full candidate pool: selecting top k before reranking loses matches.
    if not candidates:
        return {}
    # Only ranking is serialized; database retrieval can run concurrently.
    with request.app.state.reranker_lock:
        return rank_candidates(
            candidates, k=k, reranker=request.app.state.reranker
        )
