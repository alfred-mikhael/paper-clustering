"""Select paper metadata, embed papers, generate coordinates, and label clusters.

Examples::

    python -m paper_clustering.pipeline select --days-back 7 \
        --input-archive data/math_cs_metadata.zip --output-filepath data/papers.jsonl
    python -m paper_clustering.pipeline embed --input-filepath data/papers.jsonl \
        --output-path data/embedded --upload
    python -m paper_clustering.pipeline generate_coords \
        --save-umap-weights weights/umap.joblib --output-filepath data/coords.csv.gz
    python -m paper_clustering.pipeline generate_coords \
        --umap-weights weights/umap.joblib --input-filepath data/embedded \
        --metadata-filepath data/embedded --output-filepath data/coords.csv.gz
    python -m paper_clustering.pipeline cluster --input-filepath data/embedded \
        --metadata-filepath data/embedded --output-path data/clusters --seed 42

Database operations use DATABASE_URL or the standard PostgreSQL PG* variables.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gzip
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
from typing import Any

from paper_clustering.data_models import ClusterMetadata, EmbeddedPaper, PaperMetadata
from paper_clustering.interfaces import DatabaseClient

from paper_clustering.extract_metadata import (
    DEFAULT_ARCHIVE_PATH,
    read_metadata_archive,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_relevant(
    days_back: int,
    output_filepath: str | Path,
    max_results: int = 10000,
    categories: list[str] | None = None,
    keywords: list[str] | None = None,
    input_archive: str | Path = DEFAULT_ARCHIVE_PATH,
) -> bool:
    """Write complete paper metadata as JSON Lines, returning selection success.

    Dates are inclusive UTC calendar days, based on the first arXiv version.
    Match any category AND any keyword (in title/abstract); empty filters are
    unrestricted. Keep archive order and select at most ``max_results`` papers.
    A successful empty selection writes an empty file. Failures are logged and
    leave an existing output file intact.
    """
    total_started = perf_counter()
    temporary_path = None
    try:
        if days_back < 0:
            raise ValueError("days_back must be non-negative")
        output_path = Path(output_filepath)
        if output_path.resolve() == Path(input_archive).resolve():
            raise ValueError("input_archive and output_filepath must differ")
        today = datetime.now(timezone.utc).date()
        stage_started = perf_counter()
        papers = read_metadata_archive(
            input_archive,
            start_date=today - timedelta(days=days_back),
            end_date=today,
            categories=categories,
            keywords=keywords,
            max_results=max_results,
        )
        logger.info("Metadata selection took %.3fs", perf_counter() - stage_started)
        stage_started = perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            for paper in papers:
                record = asdict(paper)
                record["publication_date"] = paper.publication_date.isoformat()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
        logger.info("Metadata export took %.3fs", perf_counter() - stage_started)
        dates = [paper.publication_date.date() for paper in papers]
        logger.info(
            "Selected %d papers; earliest publication: %s; latest publication: %s; output: %s",
            len(papers),
            min(dates) if dates else "N/A",
            max(dates) if dates else "N/A",
            output_path,
        )
        return True
    except Exception:
        logger.exception("Paper selection failed")
        return False
    finally:
        logger.info("Selection finished in %.3fs", perf_counter() - total_started)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def get_month_from_batch_api():
    """Download arxiv source files from the S3 API, and process only those papers which are relevant."""
    raise NotImplementedError()


@contextmanager
def _database_client() -> Iterator[DatabaseClient]:
    """Share the configured PostgreSQL connection setup across CLI commands."""
    import psycopg
    from pgvector.psycopg import register_vector
    from paper_clustering.postgres import PostgresClient

    stage_started = perf_counter()
    with psycopg.connect(
        os.environ.get("DATABASE_URL", ""), autocommit=True, connect_timeout=10
    ) as connection:
        register_vector(connection)
        client = PostgresClient(connection)
        logger.info(
            "Database connection setup took %.3fs", perf_counter() - stage_started
        )
        yield client


def _embedding_rows(
    input_filepath: Path | None,
    metadata_filepath: Path | None,
    *,
    full_metadata: bool = False,
) -> Iterator[tuple[dict[str, Any], Any]]:
    """Read aligned metadata/vectors from batch folders, files, or the client."""
    if (input_filepath is None) != (metadata_filepath is None):
        raise ValueError("Supply both --input-filepath and --metadata-filepath")
    if input_filepath is not None and (
        input_filepath.is_dir() or metadata_filepath.is_dir()
    ):
        if not input_filepath.is_dir() or not metadata_filepath.is_dir():
            raise ValueError(
                "Input and metadata paths must both be folders or both files"
            )
        vector_batches = {
            path.name
            for path in input_filepath.iterdir()
            if path.is_dir() and re.fullmatch(r"batch_\d+", path.name)
        }
        metadata_batches = {
            path.name
            for path in metadata_filepath.iterdir()
            if path.is_dir() and re.fullmatch(r"batch_\d+", path.name)
        }
        if not vector_batches or vector_batches != metadata_batches:
            raise ValueError("Input and metadata folders must contain matching batches")
        for name in sorted(vector_batches, key=lambda name: int(name.split("_")[1])):
            yield from _embedding_rows(
                input_filepath / name / "embeddings.pt.gz",
                metadata_filepath / name / "metadata.csv.gz",
                full_metadata=full_metadata,
            )
        return

    if input_filepath is not None:
        import torch

        if input_filepath.suffix == ".gz":
            with gzip.open(input_filepath, "rb") as handle:
                vectors = torch.load(handle, map_location="cpu", weights_only=True)
        else:
            vectors = torch.load(
                input_filepath, map_location="cpu", weights_only=True, mmap=True
            )
        if (
            not isinstance(vectors, torch.Tensor)
            or vectors.layout != torch.strided
            or vectors.ndim != 2
            or vectors.shape[1] == 0
            or not vectors.is_floating_point()
        ):
            raise ValueError("Input must contain a dense 2D floating-point tensor")
        with gzip.open(metadata_filepath, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"arxiv_id", "title"}.issubset(reader.fieldnames or []):
                raise ValueError("Metadata CSV requires arxiv_id and title headers")
            count = 0
            for index, row in enumerate(reader):
                if index >= len(vectors):
                    raise ValueError("Metadata has more rows than the vector tensor")
                if not row["arxiv_id"] or row["title"] is None or None in row:
                    raise ValueError(f"Invalid metadata CSV row {index + 2}")
                count += 1
                yield row, vectors[index].detach().float().numpy()
            if count != len(vectors):
                raise ValueError("Metadata has fewer rows than the vector tensor")
        return

    with _database_client() as client:
        stage_started = perf_counter()
        records = client.select(
            table="papers",
            cols=["*"] if full_metadata else ["arxiv_id", "title", "area_embedding"],
        )
        logger.info("Database read took %.3fs", perf_counter() - stage_started)
    for record in records:
        yield record, record["area_embedding"]


def _coordinate_rows(
    input_filepath: Path | None,
    metadata_filepath: Path | None,
) -> Iterator[tuple[str, str, Any]]:
    """Select the fields needed for coordinate export from the shared reader."""
    for metadata, vector in _embedding_rows(input_filepath, metadata_filepath):
        yield metadata["arxiv_id"], metadata["title"], vector


def generate_coordinates(
    output_filepath: Path,
    *,
    umap_weights: Path | None = None,
    save_umap_weights: Path | None = None,
    input_filepath: Path | None = None,
    metadata_filepath: Path | None = None,
    batch_size: int | None = None,
) -> bool:
    """Read embeddings, generate coordinates, and write a gzipped CSV directly.

    Input paths can be batch roots or a tensor and gzipped arxiv_id,title CSV
    in matching row order. Input rows are collected in memory; coordinates
    are consumed as generate_umap yields them and written directly to the CSV.
    """
    total_started = perf_counter()
    try:
        if (input_filepath is None) != (metadata_filepath is None):
            raise ValueError("Supply both --input-filepath and --metadata-filepath")
        if umap_weights is None and save_umap_weights is None:
            raise ValueError("Fitting requires --save-umap-weights")
        output_paths = [
            path.resolve()
            for path in (output_filepath, save_umap_weights)
            if path is not None
        ]
        input_paths = [
            path.resolve()
            for path in (umap_weights, input_filepath, metadata_filepath)
            if path is not None
        ]
        if len(set(output_paths)) != len(output_paths) or set(
            output_paths
        ).intersection(input_paths):
            raise ValueError("Output paths must differ from each other and input paths")

        from paper_clustering.generate_coords import generate_umap

        stage_started = perf_counter()
        rows = list(_coordinate_rows(input_filepath, metadata_filepath))
        logger.info(
            "Coordinate input loading took %.3fs", perf_counter() - stage_started
        )
        stage_started = perf_counter()
        coordinates = generate_umap(
            (row[2] for row in rows),
            weights_path=umap_weights,
            save_weights_path=save_umap_weights,
            batch_size=batch_size,
        )
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_filepath, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("arxiv_id", "title", "umap_x", "umap_y"))
            for (arxiv_id, title, _), (x, y) in zip(rows, coordinates, strict=True):
                writer.writerow((arxiv_id, title, x, y))
        logger.info(
            "UMAP and coordinate CSV export took %.3fs", perf_counter() - stage_started
        )
        logger.info("Wrote %d coordinate rows to %s", len(rows), output_filepath)
        return True
    except Exception:
        logger.exception("Coordinate generation failed")
        return False
    finally:
        logger.info("Coordinates finished in %.3fs", perf_counter() - total_started)


def _save_embeddings(
    papers: list[EmbeddedPaper], output_path: Path, *, overwrite: bool = False
) -> None:
    """Save four gzip files; overwrite only a validated interrupted batch."""
    import numpy as np
    import torch

    output_path.mkdir(parents=True, exist_ok=overwrite)
    techniques = [technique for paper in papers for technique in paper.techniques]
    embeddings = torch.from_numpy(np.stack([paper.area_vector for paper in papers]))
    technique_vectors = (
        torch.from_numpy(np.stack([technique.embedding for technique in techniques]))
        if techniques
        else torch.empty((0, papers[0].embedding_dim))
    )
    with gzip.open(output_path / "embeddings.pt.gz", "wb") as handle:
        torch.save(embeddings, handle)
    with gzip.open(output_path / "techniques.pt.gz", "wb") as handle:
        torch.save(technique_vectors, handle)

    metadata = []
    for paper in papers:
        record = asdict(paper.metadata)
        record["authors"] = json.dumps(record["authors"], ensure_ascii=False)
        record["publication_date"] = record["publication_date"].isoformat()
        record["embedding_dim"] = paper.embedding_dim
        record["model_name"] = paper.model_name
        metadata.append(record)
    with gzip.open(
        output_path / "metadata.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    with gzip.open(
        output_path / "techniques.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("arxiv_id", "text", "score"))
        writer.writerows(
            (technique.arxiv_id, technique.text, technique.score)
            for technique in techniques
        )
    logger.info(
        "Saved %d papers and %d techniques to %s",
        len(papers),
        len(techniques),
        output_path,
    )


def generate_centroids(
    memberships_filepath: str | Path,
    coordinates_filepath: str | Path,
    output_filepath: str | Path,
    *,
    num_samples: int = 100,
    random_state: int | None = None,
) -> bool:
    """Read gzipped memberships/coordinates and write cluster_id,centroid CSV.

    Centroids are JSON arrays [x, y] inside the CSV field. Return success;
    input and calculation errors are logged before the output is opened.
    """
    try:
        from paper_clustering.cluster import approximate_centroid

        output_filepath = Path(output_filepath)
        if output_filepath.resolve() in {
            Path(memberships_filepath).resolve(),
            Path(coordinates_filepath).resolve(),
        }:
            raise ValueError("Centroid output must differ from input files")
        with gzip.open(
            memberships_filepath, "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if not {"cluster_id", "arxiv_id"}.issubset(reader.fieldnames or []):
                raise ValueError(
                    "Membership CSV requires cluster_id and arxiv_id columns"
                )
            memberships = list(reader)
        with gzip.open(
            coordinates_filepath, "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if not {"arxiv_id", "umap_x", "umap_y"}.issubset(reader.fieldnames or []):
                raise ValueError(
                    "Coordinate CSV requires arxiv_id, umap_x, umap_y columns"
                )
            coordinates = list(reader)
        centroids = approximate_centroid(
            memberships, coordinates, num_samples=num_samples, random_state=random_state
        )
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_filepath, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("cluster_id", "centroid"))
            writer.writerows((cid, json.dumps(point)) for cid, point in centroids)
        logger.info("Wrote %d cluster centroids to %s", len(centroids), output_filepath)
        return True
    except Exception:
        logger.exception("Cluster centroid generation failed")
        return False


def _save_clusters(clusters: dict[int, ClusterMetadata], filepath: Path) -> None:
    """Write the cluster hierarchy with its current labels."""
    with gzip.open(filepath, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("cluster_id", "parent_cluster_id", "size", "label"))
        writer.writerows(
            (cid, cluster.parent_cluster_id, cluster.size, cluster.label)
            for cid, cluster in clusters.items()
        )


def cluster_papers(
    output_path: Path,
    *,
    input_filepath: Path | None = None,
    metadata_filepath: Path | None = None,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
    umap_dim: int | None = None,
    server_url: str | None = None,
    model: str | None = None,
    num_samples: int = 10,
    seed: int | None = None,
    request_timeout: float = 120,
) -> bool:
    """Cluster area embeddings, label them, and save files and database records.

    File input uses the metadata exported by embed. Database input uses stored
    paper metadata; model consistency can only be checked when model_name exists.
    A labelling failure leaves the unlabelled hierarchy and memberships on disk.
    """
    total_started = perf_counter()
    try:
        import numpy as np
        from paper_clustering.cluster import (
            generate_clusters,
            label_clusters,
            validate_clustering_options,
            validate_labelling_options,
        )

        validate_clustering_options(min_cluster_size, min_samples, umap_dim)
        validate_labelling_options(num_samples, request_timeout)
        for source in (input_filepath, metadata_filepath):
            if source is not None and source.resolve() in {
                (output_path / "memberships.csv.gz").resolve(),
                (output_path / "clusters.csv.gz").resolve(),
            }:
                raise ValueError("Cluster output files must differ from input files")

        stage_started = perf_counter()
        papers = []
        for record, vector in _embedding_rows(
            input_filepath, metadata_filepath, full_metadata=True
        ):
            arxiv_id = record["arxiv_id"]
            vector = np.asarray(vector, dtype=float)
            model_name = record.get("model_name") or ""
            authors = record["authors"]
            if isinstance(authors, str):
                authors = json.loads(authors)
            publication_date = record["publication_date"]
            if not isinstance(publication_date, datetime):
                publication_date = datetime.fromisoformat(str(publication_date))
            metadata = PaperMetadata(
                authors=tuple(authors),
                publication_date=publication_date,
                arxiv_id=arxiv_id,
                title=record.get("title", ""),
                abstract=record.get("abstract", ""),
                primary_category=record["primary_category"],
                url=record["url"],
                doi=record.get("doi") or "",
            )
            papers.append(
                EmbeddedPaper(
                    metadata=metadata,
                    sections=(),
                    embedding_dim=int(record.get("embedding_dim", vector.size)),
                    model_name=model_name,
                    arxiv_id=arxiv_id,
                    area_vector=vector,
                    techniques=[],
                )
            )

        logger.info("Cluster input loading took %.3fs", perf_counter() - stage_started)
        stage_started = perf_counter()
        records, clusters = generate_clusters(
            papers,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            umap_dim=umap_dim,
        )
        logger.info("Clustering took %.3fs", perf_counter() - stage_started)
        stage_started = perf_counter()
        output_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(
            output_path / "memberships.csv.gz", "wt", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("cluster_id", "arxiv_id", "title", "certainty"))
            writer.writerows(
                (record.cluster_id, record.arxiv_id, record.title, record.certainty)
                for record in records
            )
        _save_clusters(clusters, output_path / "clusters.csv.gz")
        logger.info(
            "Unlabelled cluster export took %.3fs", perf_counter() - stage_started
        )
        logger.info("Saved %d clusters for %d papers", len(clusters), len(papers))
        if clusters:
            stage_started = perf_counter()
            labelled = label_clusters(
                records,
                clusters,
                [paper.metadata for paper in papers],
                random_state=seed,
                server_url=server_url,
                model=model,
                request_timeout=request_timeout,
                num_samples=num_samples,
            )
            logger.info("Cluster labelling took %.3fs", perf_counter() - stage_started)
            stage_started = perf_counter()
            _save_clusters(labelled, output_path / "clusters.csv.gz")
            logger.info(
                "Labelled cluster export took %.3fs", perf_counter() - stage_started
            )
            logger.info("Labelled %d clusters", len(labelled))
            stage_started = perf_counter()
            with _database_client() as client:
                with client.transaction():
                    if not client.insert(
                        table="cluster_metadata",
                        records=[
                            {"id": cid, **asdict(cluster)}
                            for cid, cluster in labelled.items()
                        ],
                    ):
                        raise RuntimeError("Cluster metadata insertion failed")
                    if not client.insert(
                        table="clusters",
                        records=[asdict(record) for record in records],
                    ):
                        raise RuntimeError("Cluster membership insertion failed")
            logger.info(
                "Cluster database upload took %.3fs", perf_counter() - stage_started
            )
        return True
    except Exception:
        logger.exception("Clustering or cluster labelling failed")
        return False
    finally:
        logger.info(
            "Clustering pipeline finished in %.3fs", perf_counter() - total_started
        )


def _validate_embedding_options(
    *,
    upload: bool,
    output_path: Path | None,
    batch_size: int,
    merge_threshold: float,
    technique_k: int,
    pooling: str | None,
    technique_batch_size: int = 32,
    embedding_batch_size: int | None = None,
) -> None:
    """Validate the embedding workflow independently of argument parsing."""
    if not upload and output_path is None:
        raise ValueError("embed requires --upload, --output-path, or both")
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if technique_batch_size < 1:
        raise ValueError("--technique-batch-size must be positive")
    if embedding_batch_size is not None and embedding_batch_size < 1:
        raise ValueError("--embedding-batch-size must be positive")
    if not -1 <= merge_threshold <= 1:
        raise ValueError("--merge-threshold must be between -1 and 1")
    if technique_k < 0:
        raise ValueError("--technique-k must be non-negative")
    if pooling not in (None, "mean", "max"):
        raise ValueError("--pooling must be either None, 'mean' or 'max'")


def _write_batch_manifest(batch_path: Path, manifest: dict[str, Any]) -> None:
    """Replace the manifest atomically so interruption preserves the old status."""
    pending = batch_path / "manifest.json.tmp"
    pending.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pending.replace(batch_path / "manifest.json")


def _resume_embedding_offset(
    ids: list[str],
    output_path: Path | None,
    settings: dict[str, Any],
    resume: bool,
) -> int:
    """Validate saved batches and finish pending upserts before loading models."""
    if output_path is None:
        if resume:
            raise ValueError("--resume requires --output-path")
        return 0
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("--output-path must be a directory")
    batches = sorted(output_path.glob("batch_*"))
    if batches and not resume:
        raise ValueError(
            "Output already contains batches; use --resume or a new output path"
        )
    offset = 0
    for number, batch_path in enumerate(batches, start=1):
        if batch_path.name != f"batch_{number:04d}" or not batch_path.is_dir():
            raise ValueError(f"Expected consecutive batch folders; found {batch_path}")
        manifest_path = batch_path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing manifest: {manifest_path}; cannot safely resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Older exports used a fixed 500-paper batch size.
        manifest.setdefault("batch_size", 500)
        expected_ids = ids[offset : offset + settings["batch_size"]]
        if not expected_ids or manifest.get("arxiv_ids") != expected_ids:
            raise ValueError(f"Input paper IDs differ from {manifest_path}")
        for key, value in settings.items():
            if manifest.get(key) != value or key not in manifest:
                raise ValueError(f"{key} differs from {manifest_path}")
        status = manifest.get("status")
        if status == "processing":
            if batch_path != batches[-1]:
                raise ValueError(
                    f"Incomplete batch precedes later batches: {batch_path}"
                )
            logger.info("Recomputing interrupted batch %s", batch_path)
            return offset
        if status not in ("completed", "uploading"):
            raise ValueError(f"Unknown batch status in {manifest_path}: {status!r}")
        for filename in (
            "embeddings.pt.gz",
            "techniques.pt.gz",
            "metadata.csv.gz",
            "techniques.csv.gz",
        ):
            if not (batch_path / filename).is_file():
                raise ValueError(f"Saved batch is missing {batch_path / filename}")
        if status == "uploading":
            if not settings["upload"] or batch_path != batches[-1]:
                raise ValueError(f"Invalid pending upload batch: {batch_path}")
            logger.info("Retrying saved batch upload: %s", batch_path)
            papers = _load_saved_embeddings(batch_path)
            saved_ids = [paper.arxiv_id for paper in papers]
            expected_id_set = set(expected_ids)
            # API metadata may resolve an unversioned request to a versioned ID.
            # Explicitly requested versions must still match exactly.
            if len(set(saved_ids)) != len(saved_ids) or any(
                saved_id not in expected_id_set
                and re.sub(r"v[0-9]+$", "", saved_id) not in expected_id_set
                for saved_id in saved_ids
            ):
                raise ValueError(f"Saved paper IDs do not match {manifest_path}")
            _upload_embedding_batch(papers)
            manifest["status"] = "completed"
            _write_batch_manifest(batch_path, manifest)
        offset += len(expected_ids)
        logger.info(
            "Skipping completed batch %s (%d papers)", batch_path, len(expected_ids)
        )
    return offset


def _load_saved_embeddings(batch_path: Path) -> list[EmbeddedPaper]:
    """Restore a batch's paper and technique vectors for upload replay."""
    import torch
    from paper_clustering.data_models import Technique

    papers = {}
    for record, vector in _embedding_rows(
        batch_path / "embeddings.pt.gz",
        batch_path / "metadata.csv.gz",
        full_metadata=True,
    ):
        embedding_dim = int(record.pop("embedding_dim"))
        model_name = record.pop("model_name")
        record["authors"] = tuple(json.loads(record["authors"]))
        record["publication_date"] = datetime.fromisoformat(record["publication_date"])
        metadata = PaperMetadata(**record)
        if metadata.arxiv_id in papers or len(vector) != embedding_dim:
            raise ValueError(f"Invalid paper metadata in {batch_path}")
        papers[metadata.arxiv_id] = EmbeddedPaper(
            metadata=metadata,
            sections=(),
            embedding_dim=embedding_dim,
            model_name=model_name,
            arxiv_id=metadata.arxiv_id,
            area_vector=vector,
            techniques=[],
        )
    with gzip.open(batch_path / "techniques.pt.gz", "rb") as handle:
        vectors = torch.load(handle, map_location="cpu", weights_only=True)
    if (
        not isinstance(vectors, torch.Tensor)
        or vectors.ndim != 2
        or not vectors.is_floating_point()
    ):
        raise ValueError(f"Invalid technique vectors in {batch_path}")
    with gzip.open(
        batch_path / "techniques.csv.gz", "rt", encoding="utf-8", newline=""
    ) as handle:
        for record, vector in zip(csv.DictReader(handle), vectors, strict=True):
            paper = papers[record["arxiv_id"]]
            if len(vector) != paper.embedding_dim:
                raise ValueError(f"Invalid technique dimension in {batch_path}")
            paper.techniques.append(
                Technique(
                    arxiv_id=paper.arxiv_id,
                    text=record["text"],
                    embedding=vector.numpy(),
                    score=float(record["score"]),
                )
            )
    return list(papers.values())


def _upload_embedding_batch(papers: list[EmbeddedPaper]) -> None:
    """Use the existing transactional upserts for new and resumed batches."""
    from paper_clustering.upload import upload

    with _database_client() as client:
        stage_started = perf_counter()
        if upload(papers, client) != (True, True):
            raise RuntimeError("Upload did not complete successfully")
        logger.info("Database upload took %.3fs", perf_counter() - stage_started)


def embed_papers(
    input_filepath: Path | None = None,
    *,
    ids_filepath: Path | None = None,
    upload: bool = False,
    output_path: Path | None = None,
    resume: bool = False,
    classifier_weights: Path = Path("weights/technique_classifier_checkpoint.pt"),
    embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe",
    device: str | None = None,
    batch_size: int = 500,
    technique_batch_size: int = 32,
    embedding_batch_size: int | None = None,
    include_proofs: bool = False,
    merge_threshold: float = 0.9,
    technique_k: int = 3,
    pooling: str | None = None,
) -> bool:
    """Embed JSONL metadata or newline-separated IDs, optionally resuming an export."""
    total_started = perf_counter()
    try:
        if (input_filepath is None) == (ids_filepath is None):
            raise ValueError("Supply exactly one of --input-filepath or --ids-filepath")
        _validate_embedding_options(
            upload=upload,
            output_path=output_path,
            batch_size=batch_size,
            merge_threshold=merge_threshold,
            technique_k=technique_k,
            pooling=pooling,
            technique_batch_size=technique_batch_size,
            embedding_batch_size=embedding_batch_size,
        )
        embedding_batch_size = (
            8 if embedding_batch_size is None else embedding_batch_size
        )
        stage_started = perf_counter()
        metadata_by_id = {}
        source_path = input_filepath if input_filepath is not None else ids_filepath
        with source_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if ids_filepath is not None:
                    metadata_by_id.setdefault(line.strip(), None)
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Expected a paper metadata object")
                    if not isinstance(record.get("authors"), list) or not all(
                        isinstance(author, str) for author in record["authors"]
                    ):
                        raise ValueError("authors must be a list of strings")
                    record["authors"] = tuple(record["authors"])
                    record["publication_date"] = datetime.fromisoformat(
                        record["publication_date"]
                    )
                    metadata = PaperMetadata(**record)
                    if not isinstance(metadata.arxiv_id, str) or not metadata.arxiv_id:
                        raise ValueError("arxiv_id must be a nonempty string")
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(
                        f"Invalid paper metadata at line {line_number}; "
                        "use the JSON Lines output from select"
                    ) from error
                metadata_by_id.setdefault(metadata.arxiv_id, metadata)
        ids = list(metadata_by_id)
        logger.info("Metadata loading took %.3fs", perf_counter() - stage_started)
        settings = {
            "batch_size": batch_size,
            "embedding_model": embedding_model,
            "classifier_weights": str(classifier_weights.resolve()),
            "pooling": pooling,
            "metadata_filename": "metadata.csv.gz",
            "include_proofs": include_proofs,
            "merge_threshold": merge_threshold,
            "technique_k": technique_k,
            "upload": upload,
        }
        start_offset = _resume_embedding_offset(ids, output_path, settings, resume)
        if not ids:
            logger.info("No arXiv IDs to embed")
            return True
        if start_offset == len(ids):
            logger.info("All requested papers are already complete")
            return True
        # Keep model/database dependencies out of archive selection and CLI help.
        import numpy as np
        from tqdm import tqdm
        import requests
        from sentence_transformers import SentenceTransformer

        from paper_clustering.embedding import _choose_device, embed
        from paper_clustering.extract_techniques import (
            extract_techniques_and_embed_batch,
        )
        from paper_clustering.extract_text import get_paper
        from paper_clustering.technique_classifier import SciBERTClassifier

        device = _choose_device(device)
        stage_started = perf_counter()
        classifier = SciBERTClassifier(weights_path=classifier_weights).to(device)
        logger.info("Classifier loading took %.3fs", perf_counter() - stage_started)
        stage_started = perf_counter()
        encoder = SentenceTransformer(
            embedding_model, trust_remote_code=True, device=device
        )
        logger.info(
            "Embedding model loading took %.3fs", perf_counter() - stage_started
        )
        embedded_count = start_offset
        with requests.Session() as session:
            for offset in range(start_offset, len(ids), batch_size):
                batch_started = perf_counter()
                batch_ids = ids[offset : offset + batch_size]
                batch_path = (
                    output_path / f"batch_{offset // batch_size + 1:04d}"
                    if output_path is not None
                    else None
                )
                manifest = {**settings, "arxiv_ids": batch_ids, "status": "processing"}
                if batch_path is not None:
                    batch_path.mkdir(parents=True, exist_ok=True)
                    _write_batch_manifest(batch_path, manifest)
                logger.info(
                    "Processing batch %d: %d papers",
                    offset // batch_size + 1,
                    len(batch_ids),
                )
                stage_started = perf_counter()
                papers = []
                for index, arxiv_id in tqdm(
                    enumerate(batch_ids, start=offset + 1), total=len(batch_ids)
                ):
                    logger.info(
                        "Downloading paper %d/%d: %s", index, len(ids), arxiv_id
                    )
                    paper = get_paper(
                        session,
                        arxiv_id,
                        metadata=metadata_by_id[arxiv_id],
                        include_proofs=include_proofs,
                        retries=1,
                    )
                    if not any(section.text for section in paper.sections):
                        logger.warning("No usable source paragraphs for %s", arxiv_id)
                    papers.append(paper)
                logger.info(
                    "Paper download/extraction took %.3fs",
                    perf_counter() - stage_started,
                )
                stage_started = perf_counter()
                area_vectors = embed(
                    [
                        f"{paper.metadata.title}\n\n{paper.metadata.abstract}"
                        for paper in papers
                    ],
                    encoder,
                    batch_size=embedding_batch_size,
                    prompt="clustering: ",
                    device=device,
                )
                logger.info("Area embedding took %.3fs", perf_counter() - stage_started)
                stage_started = perf_counter()
                techniques = extract_techniques_and_embed_batch(
                    papers,
                    classifier,
                    encoder,
                    threshold=merge_threshold,
                    k=technique_k,
                    pooling=pooling,
                    technique_batch_size=technique_batch_size,
                    embedding_batch_size=embedding_batch_size,
                )
                logger.info(
                    "Technique extraction/embedding took %.3fs",
                    perf_counter() - stage_started,
                )
                stage_started = perf_counter()
                embedded_papers = [
                    EmbeddedPaper(
                        metadata=paper.metadata,
                        sections=paper.sections,
                        embedding_dim=len(vector),
                        model_name=embedding_model,
                        arxiv_id=paper.metadata.arxiv_id,
                        area_vector=np.asarray(vector),
                        techniques=passages,
                    )
                    for paper, vector, passages in zip(
                        papers, area_vectors, techniques, strict=True
                    )
                ]
                logger.info(
                    "Embedding record preparation took %.3fs",
                    perf_counter() - stage_started,
                )
                if output_path is not None:
                    stage_started = perf_counter()
                    _save_embeddings(
                        embedded_papers,
                        batch_path,
                        overwrite=True,
                    )
                    logger.info(
                        "Batch file export took %.3fs", perf_counter() - stage_started
                    )
                if upload:
                    if batch_path is not None:
                        manifest["status"] = "uploading"
                        _write_batch_manifest(batch_path, manifest)
                    _upload_embedding_batch(embedded_papers)
                if batch_path is not None:
                    manifest["status"] = "completed"
                    _write_batch_manifest(batch_path, manifest)
                logger.info(
                    "Embedding batch %d took %.3fs",
                    offset // batch_size + 1,
                    perf_counter() - batch_started,
                )
                embedded_count += len(embedded_papers)
                logger.info("Embedded %d/%d papers", embedded_count, len(ids))
                del papers, area_vectors, techniques
                del embedded_papers
        return True
    except Exception:
        logger.exception("Pipeline embedding failed")
        return False
    finally:
        logger.info(
            "Embedding pipeline finished in %.3fs", perf_counter() - total_started
        )


def main(argv: list[str] | None = None) -> int:
    """Run selection, embedding, coordinates, or labelled clustering.

    Embed reads JSON Lines metadata from select or a file with one arXiv ID per line.
    Duplicate IDs keep their first record. --batch-size controls papers per batch.
    With --upload, commit each batch using the existing upload function. With
    --output-path, save each batch to a new batch_0001, batch_0002, ... folder.
    A failure stops subsequent batches; prior uploads and batch files remain.
    With --resume, validate manifests and skip completed batches.
    Generate visualization coordinates separately.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Select paper metadata from a ZIP")
    select.add_argument("--days-back", type=int, required=True)
    select.add_argument("--input-archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    select.add_argument("--output-filepath", type=Path, required=True)
    select.add_argument("--max-results", type=int, default=10000)
    select.add_argument("--categories", nargs="+")
    select.add_argument("--keywords", nargs="+")

    embed_parser = commands.add_parser(
        "embed", help="Embed papers for file export or upload"
    )
    embed_input = embed_parser.add_mutually_exclusive_group(required=True)
    embed_input.add_argument(
        "--input-filepath",
        type=Path,
        help="Paper metadata JSON Lines file produced by select",
    )
    embed_input.add_argument(
        "--ids-filepath",
        type=Path,
        help="Text file with one arXiv ID per line; metadata is fetched automatically",
    )
    embed_parser.add_argument(
        "--upload", action="store_true", help="Upload to the database"
    )
    embed_parser.add_argument(
        "--output-path",
        type=Path,
        help="Root for batch folders containing gzipped tensors and CSV metadata",
    )
    embed_parser.add_argument(
        "--classifier-weights",
        type=Path,
        default=Path("weights/technique_classifier_checkpoint.pt"),
    )
    embed_parser.add_argument(
        "--embedding-model", default="nomic-ai/nomic-embed-text-v2-moe"
    )
    embed_parser.add_argument(
        "--resume", action="store_true", help="Continue batches saved to --output-path"
    )
    embed_parser.add_argument("--device", help="Torch device; default: automatic")
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Papers per processing/export batch (default: 500)",
    )
    embed_parser.add_argument(
        "--technique-batch-size",
        type=int,
        default=32,
        help="Technique classifier batch size (default: 32)",
    )
    embed_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        help="Area and technique embedding batch size (default: 8)",
    )
    embed_parser.add_argument("--include-proofs", action="store_true", default=True)
    embed_parser.add_argument("--merge-threshold", type=float, default=0.9)
    embed_parser.add_argument("--technique-k", type=int, default=3)
    embed_parser.add_argument("--pooling", type=str, default="max")
    coords_parser = commands.add_parser(
        "generate_coords", help="Fit/save UMAP or apply a saved reducer"
    )
    coords_parser.add_argument(
        "--umap-weights", type=Path, help="Existing joblib reducer"
    )
    coords_parser.add_argument(
        "--save-umap-weights", type=Path, help="Required model destination when fitting"
    )
    coords_parser.add_argument(
        "--input-filepath",
        type=Path,
        help="Batch root containing batch_*/embeddings.pt.gz, or a .pt/.pt.gz file; default: database",
    )
    coords_parser.add_argument(
        "--metadata-filepath",
        type=Path,
        help="Matching batch root containing batch_*/metadata.csv.gz, or a gzipped CSV",
    )
    coords_parser.add_argument(
        "--output-filepath",
        type=Path,
        required=True,
        help="Gzipped CSV destination with arxiv_id,title,umap_x,umap_y columns",
    )
    coords_parser.add_argument(
        "--batch-size",
        type=int,
        help="Transform batch size (default: 1000); requires --umap-weights",
    )
    cluster_parser = commands.add_parser(
        "cluster", help="Cluster area vectors and label them"
    )
    cluster_parser.add_argument(
        "--input-filepath",
        type=Path,
        help="Batch root or .pt/.pt.gz tensor; omit both input paths for the database",
    )
    cluster_parser.add_argument(
        "--metadata-filepath",
        type=Path,
        help="Matching batch root or exported metadata.csv.gz, including abstracts",
    )
    cluster_parser.add_argument("--output-path", type=Path, required=True)
    cluster_parser.add_argument("--min-cluster-size", type=int, default=5)
    cluster_parser.add_argument("--min-samples", type=int)
    cluster_parser.add_argument("--metric", default="euclidean")
    cluster_parser.add_argument("--umap-dim", type=int)
    cluster_parser.add_argument(
        "--server-url",
        help="Labelling server; default: LLAMA_CPP_SERVER_URL or localhost:8080",
    )
    cluster_parser.add_argument("--model", help="Optional labelling model name")
    cluster_parser.add_argument("--num-samples", type=int, default=10)
    cluster_parser.add_argument("--seed", type=int, help="Seed for label sampling")
    cluster_parser.add_argument("--request-timeout", type=float, default=120)
    centroids_parser = commands.add_parser(
        "generate_centroids",
        help="Estimate cluster centers from sampled UMAP coordinates",
    )
    centroids_parser.add_argument("--memberships-filepath", type=Path, required=True)
    centroids_parser.add_argument("--coordinates-filepath", type=Path, required=True)
    centroids_parser.add_argument("--output-filepath", type=Path, required=True)
    centroids_parser.add_argument("--num-samples", type=int, default=100)
    centroids_parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("psycopg").setLevel(logging.DEBUG)

    if args.command == "generate_centroids":
        return (
            0
            if generate_centroids(
                args.memberships_filepath,
                args.coordinates_filepath,
                args.output_filepath,
                num_samples=args.num_samples,
                random_state=args.seed,
            )
            else 1
        )

    if args.command == "select":
        return (
            0
            if get_relevant(
                days_back=args.days_back,
                output_filepath=args.output_filepath,
                max_results=args.max_results,
                categories=args.categories,
                keywords=args.keywords,
                input_archive=args.input_archive,
            )
            else 1
        )

    if args.command == "generate_coords":
        return (
            0
            if generate_coordinates(
                output_filepath=args.output_filepath,
                umap_weights=args.umap_weights,
                save_umap_weights=args.save_umap_weights,
                input_filepath=args.input_filepath,
                metadata_filepath=args.metadata_filepath,
                batch_size=args.batch_size,
            )
            else 1
        )

    if args.command == "cluster":
        return (
            0
            if cluster_papers(
                args.output_path,
                input_filepath=args.input_filepath,
                metadata_filepath=args.metadata_filepath,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                metric=args.metric,
                umap_dim=args.umap_dim,
                server_url=args.server_url,
                model=args.model,
                num_samples=args.num_samples,
                seed=args.seed,
                request_timeout=args.request_timeout,
            )
            else 1
        )

    return (
        0
        if embed_papers(
            args.input_filepath,
            ids_filepath=args.ids_filepath,
            upload=args.upload,
            output_path=args.output_path,
            resume=args.resume,
            technique_batch_size=args.technique_batch_size,
            embedding_batch_size=args.embedding_batch_size,
            classifier_weights=args.classifier_weights,
            embedding_model=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
            include_proofs=args.include_proofs,
            merge_threshold=args.merge_threshold,
            technique_k=args.technique_k,
            pooling=args.pooling,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
