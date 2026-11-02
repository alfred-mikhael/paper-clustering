"""Local pipeline artifacts, their formats, and file-level validation."""

from collections.abc import Iterable, Iterator
import csv
from dataclasses import asdict
from datetime import datetime
import gzip
import json
import logging
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import quote

from paper_clustering.data_models import (
    ArxivSection,
    ClusterMetadata,
    ClusterRecord,
    EmbeddedPaper,
    Paper,
    PaperMetadata,
)

logger = logging.getLogger(__name__)


class LocalStorage:
    """Read and write local artifacts without processing or recovery policy.

    Paths are explicit; instances retain no current directory or batch state.
    Database access and decisions to skip, recompute, or upload remain with
    the caller. Tensor dependencies are loaded only when tensor files are used.
    """

    def load_paper_input(
        self,
        input_filepath: Path | None, ids_filepath: Path | None
    ) -> dict[str, PaperMetadata | None]:
        """Read the shared selection/ID input, preserving order and deduplicating IDs."""
        if (input_filepath is None) == (ids_filepath is None):
            raise ValueError("Supply exactly one of --input-filepath or --ids-filepath")
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
                    if isinstance(record.get("arxiv_id"), str):
                        record["arxiv_id"] = re.sub(r"v[0-9]+$", "", record["arxiv_id"])
                    metadata = PaperMetadata(**record)
                    if not isinstance(metadata.arxiv_id, str) or not metadata.arxiv_id:
                        raise ValueError("arxiv_id must be a nonempty string")
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(
                        f"Invalid paper metadata at line {line_number}; "
                        "use the JSON Lines output from select"
                    ) from error
                metadata_by_id.setdefault(metadata.arxiv_id, metadata)
        return metadata_by_id

    def save_paper_metadata(self, filepath: Path, papers: Iterable[PaperMetadata]) -> None:
        """Write selection JSONL, preserving an existing file if serialization fails."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=filepath.parent, delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                for paper in papers:
                    record = asdict(paper)
                    record["publication_date"] = paper.publication_date.isoformat()
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary_path.replace(filepath)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def source_filepath(self, source_path: Path, arxiv_id: str) -> Path:
        """Encode IDs, including old-style IDs containing slashes, as filenames."""
        return source_path / f"{quote(arxiv_id, safe='')}.json.gz"

    def load_source_paper(
        self, source_path: Path, arxiv_id: str, include_proofs: bool
    ) -> Paper:
        """Restore downloaded text and reject incompatible extraction settings."""
        filepath = self.source_filepath(source_path, arxiv_id)
        with gzip.open(filepath, "rt", encoding="utf-8") as handle:
            record = json.load(handle)
        if record["requested_arxiv_id"] != arxiv_id:
            raise ValueError(f"Paper ID differs from {filepath}")
        if record["include_proofs"] != include_proofs:
            raise ValueError(f"--include-proofs differs from {filepath}")
        metadata = record["metadata"]
        metadata["authors"] = tuple(metadata["authors"])
        metadata["publication_date"] = datetime.fromisoformat(metadata["publication_date"])
        return Paper(
            metadata=PaperMetadata(**metadata),
            sections=tuple(
                ArxivSection(**{**section, "text": tuple(section["text"])})
                for section in record["sections"]
            ),
        )

    def save_source_paper(
        self,
        source_path: Path,
        paper: Paper,
        *,
        requested_id: str,
        include_proofs: bool,
    ) -> None:
        """Save a downloaded paper using the existing source-cache format."""
        source_path.mkdir(parents=True, exist_ok=True)
        filepath = self.source_filepath(source_path, requested_id)
        record = asdict(paper)
        record["metadata"]["publication_date"] = paper.metadata.publication_date.isoformat()
        record.update(requested_arxiv_id=requested_id, include_proofs=include_proofs)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=source_path, delete=False) as handle:
                temporary_path = Path(handle.name)
            with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
            temporary_path.replace(filepath)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def iter_embedding_rows(
        self,
        input_filepath: Path,
        metadata_filepath: Path,
    ) -> Iterator[tuple[dict[str, Any], Any]]:
        """Read aligned metadata/vectors from batch folders or individual files."""
        if input_filepath.is_dir() or metadata_filepath.is_dir():
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
                yield from self.iter_embedding_rows(
                    input_filepath / name / "embeddings.pt.gz",
                    metadata_filepath / name / "metadata.csv.gz",
                )
            return

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

    def load_embedding_batch(self, batch_path: Path) -> list[EmbeddedPaper]:
        """Restore a batch's paper and technique vectors for upload replay."""
        import torch
        from paper_clustering.data_models import Technique

        papers = {}
        for record, vector in self.iter_embedding_rows(
            batch_path / "embeddings.pt.gz",
            batch_path / "metadata.csv.gz",
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

    def save_embedding_batch(
        self,
        output_path: Path, papers: list[EmbeddedPaper], *, overwrite: bool = False
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

    def validate_embedding_batch_files(self, batch_path: Path) -> None:
        """Check that all four files required by a completed batch are present."""
        for filename in (
            "embeddings.pt.gz",
            "techniques.pt.gz",
            "metadata.csv.gz",
            "techniques.csv.gz",
        ):
            if not (batch_path / filename).is_file():
                raise ValueError(f"Saved batch is missing {batch_path / filename}")

    def read_manifest(self, batch_path: Path) -> dict[str, Any]:
        """Read a batch manifest; interpretation of its status belongs to the pipeline."""
        filepath = batch_path / "manifest.json"
        if not filepath.is_file():
            raise ValueError(f"Missing manifest: {filepath}; cannot safely resume")
        return json.loads(filepath.read_text(encoding="utf-8"))

    def write_manifest(self, batch_path: Path, manifest: dict[str, Any]) -> None:
        """Replace the manifest atomically so interruption preserves the old status."""
        pending = batch_path / "manifest.json.tmp"
        pending.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        pending.replace(batch_path / "manifest.json")

    def load_memberships(self, filepath: str | Path) -> list[dict[str, str]]:
        """Read membership rows for centroid calculation."""
        return self._read_csv(filepath, ("cluster_id", "arxiv_id"))

    def load_coordinates(self, filepath: str | Path) -> list[dict[str, str]]:
        """Read coordinate rows for centroid calculation."""
        return self._read_csv(filepath, ("arxiv_id", "umap_x", "umap_y"))

    def save_coordinates(
        self,
        filepath: Path,
        papers: Iterable[tuple[str, str]],
        coordinates: Iterable[tuple[float, float]],
    ) -> None:
        """Stream aligned paper identities and coordinates to a gzipped CSV."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(filepath, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("arxiv_id", "title", "umap_x", "umap_y"))
            for (arxiv_id, title), (x, y) in zip(papers, coordinates, strict=True):
                writer.writerow((arxiv_id, title, x, y))

    def save_centroids(
        self, filepath: Path, centroids: Iterable[tuple[int, tuple[float, float]]]
    ) -> None:
        """Write centroids as JSON arrays inside a gzipped CSV."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(filepath, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("cluster_id", "centroid"))
            writer.writerows((cid, json.dumps(point)) for cid, point in centroids)

    def load_clusters(
        self, output_path: Path
    ) -> tuple[list[ClusterRecord], dict[int, ClusterMetadata]]:
        """Restore memberships and their cluster hierarchy from an existing export."""
        records = [
            ClusterRecord(
                cluster_id=int(row["cluster_id"]),
                arxiv_id=row["arxiv_id"],
                title=row["title"],
                certainty=float(row["certainty"]),
            )
            for row in self._read_csv(
                output_path / "memberships.csv.gz",
                ("cluster_id", "arxiv_id", "title", "certainty"),
            )
        ]
        clusters = {
            int(row["cluster_id"]): ClusterMetadata(
                parent_cluster_id=int(row["parent_cluster_id"]),
                size=int(row["size"]),
                label=row["label"],
            )
            for row in self._read_csv(
                output_path / "clusters.csv.gz",
                ("cluster_id", "parent_cluster_id", "size", "label"),
            )
        }
        return records, clusters

    def save_clusters(
        self,
        output_path: Path,
        memberships: Iterable[ClusterRecord],
        clusters: dict[int, ClusterMetadata],
    ) -> None:
        """Write memberships and hierarchy using the existing two-file format."""
        output_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(
            output_path / "memberships.csv.gz", "wt", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("cluster_id", "arxiv_id", "title", "certainty"))
            writer.writerows(
                (record.cluster_id, record.arxiv_id, record.title, record.certainty)
                for record in memberships
            )
        self.save_cluster_metadata(output_path / "clusters.csv.gz", clusters)

    def save_cluster_metadata(
        self,
        filepath: Path,
        clusters: dict[int, ClusterMetadata],
        *,
        checkpoint: bool = False,
    ) -> None:
        """Update labels, preserving the existing label-progress checkpoint behavior.

        Full exports write directly. A progress checkpoint replaces the previous
        hierarchy only after the complete updated hierarchy has been written.
        """
        if not checkpoint:
            self._write_cluster_metadata(filepath, clusters)
            return
        pending_path = filepath.with_name(filepath.name + ".tmp")
        try:
            self._write_cluster_metadata(pending_path, clusters)
            pending_path.replace(filepath)
        finally:
            pending_path.unlink(missing_ok=True)

    def _write_cluster_metadata(
        self, filepath: Path, clusters: dict[int, ClusterMetadata]
    ) -> None:
        """Write the cluster hierarchy with its current labels."""
        with gzip.open(filepath, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("cluster_id", "parent_cluster_id", "size", "label"))
            writer.writerows(
                (cid, cluster.parent_cluster_id, cluster.size, cluster.label)
                for cid, cluster in clusters.items()
            )

    def _read_csv(
        self, filepath: str | Path, required_columns: tuple[str, ...]
    ) -> list[dict[str, str]]:
        with gzip.open(filepath, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(required_columns).issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{filepath}: CSV requires {', '.join(required_columns)} columns"
                )
            return list(reader)
