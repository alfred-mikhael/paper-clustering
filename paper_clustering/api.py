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
from time import perf_counter
from typing import Annotated

import psycopg
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

from paper_clustering.postgres import PostgresClient
from paper_clustering.query import SearchResult, rank_candidates, retrieve_candidates
from paper_clustering.reranker import SLMReranker
from paper_clustering.interfaces import DatabaseClient

# Inherit Uvicorn's configured INFO handler when running the documented command.
logger = logging.getLogger("uvicorn.error").getChild(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources at startup and release them at shutdown."""
    startup_started = perf_counter()
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
        stage_started = perf_counter()
        pool.open(wait=True, timeout=10)
        logger.info("Database pool startup took %.3fs", perf_counter() - stage_started)
        app.state.database_pool = pool
        stage_started = perf_counter()
        app.state.reranker = SLMReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
        logger.info("Reranker loading took %.3fs", perf_counter() - stage_started)
        app.state.reranker_lock = Lock()
        logger.info("API startup took %.3fs", perf_counter() - startup_started)
        yield
    finally:
        stage_started = perf_counter()
        pool.close()
        # Drop the application's model reference on shutdown.
        if hasattr(app.state, "reranker"):
            del app.state.reranker
        logger.info("API shutdown took %.3fs", perf_counter() - stage_started)


app = FastAPI(title="Paper Technique Search", lifespan=lifespan)

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    """Include routing, endpoint work, and response preparation in request timing."""
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        logger.info(
            "%s %s status=%d took %.3fs",
            request.method,
            request.url.path,
            status_code,
            perf_counter() - started,
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
)

@contextmanager
def get_database_client(request: Request) -> Iterator[DatabaseClient]:
    """Lend a pooled connection only for the caller's database operations."""
    started = perf_counter()
    try:
        # The client is lightweight and request-local; its physical connection
        # is reused. The context returns it even if retrieval raises an error.
        with request.app.state.database_pool.connection() as connection:
            logger.info("Database connection wait took %.3fs", perf_counter() - started)
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
    rerank: Annotated[bool, Query(description="Order papers using the reranker")] = True,
) -> dict[str, list[SearchResult]]:
    """Find up to k similar papers, grouped by target arXiv ID.

    Papers are ordered by their best reranker score by default. Set rerank=false
    to order papers by their best vector similarity instead. The reranker stays
    loaded for other requests. Each match's similarity remains the original
    vector similarity. An empty object
    means no matches were found, including when the query has no stored passages.
    """
    # FastAPI returns HTTP 422 for invalid query parameters.
    # A synchronous endpoint runs in a worker thread so blocking database calls
    # do not block the async event loop. Dataclasses are serialized to JSON using
    # the response_model above, which also documents the result fields in /docs.
    total_started = perf_counter()
    try:
        with get_database_client(request) as client:
            stage_started = perf_counter()
            try:
                candidates = retrieve_candidates(arxiv_id.strip(), client, k=k)
            finally:
                logger.info(
                    "Query %s retrieval took %.3fs",
                    arxiv_id,
                    perf_counter() - stage_started,
                )
        # Return the connection before waiting for the GPU; only ranking is serialized.
        if not candidates:
            return {}
        if not rerank:
            return rank_candidates(candidates, k=k)
        stage_started = perf_counter()
        with request.app.state.reranker_lock:
            logger.info(
                "Query %s reranker lock wait took %.3fs",
                arxiv_id,
                perf_counter() - stage_started,
            )
            stage_started = perf_counter()
            try:
                return rank_candidates(
                    candidates, k=k, reranker=request.app.state.reranker
                )
            finally:
                logger.info(
                    "Query %s reranking took %.3fs",
                    arxiv_id,
                    perf_counter() - stage_started,
                )
    finally:
        logger.info(
            "Query %s total took %.3fs", arxiv_id, perf_counter() - total_started
        )
