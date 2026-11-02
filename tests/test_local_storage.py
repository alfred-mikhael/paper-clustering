"""Local artifact compatibility checks; run with unittest discover -s tests."""

import csv
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from paper_clustering.data_models import (
    ArxivSection,
    ClusterMetadata,
    ClusterRecord,
    EmbeddedPaper,
    Paper,
    PaperMetadata,
    Technique,
)
from paper_clustering.local_storage import LocalStorage


class LocalStorageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.storage = LocalStorage()
        self.metadata = PaperMetadata(
            authors=("Author, One", "Zoë"),
            publication_date=datetime(2026, 1, 2, tzinfo=timezone.utc),
            arxiv_id="math/1234567",
            title='A title, with "quotes"',
            abstract="First paragraph.\nSecond paragraph.",
            primary_category="math.CO",
            url="https://arxiv.org/abs/math/1234567",
            doi="",
        )
        self.paper = Paper(
            metadata=self.metadata,
            sections=(ArxivSection(
                arxiv_id=self.metadata.arxiv_id,
                section_index=0,
                section_header="Introduction",
                major_section_header="Introduction",
                text=("A paragraph.", "Another paragraph."),
                raw_latex=r"\section{Introduction}",
            ),),
        )

    def test_metadata_round_trip_and_version_deduplication(self):
        path = self.root / "papers.jsonl"
        versioned = replace(self.metadata, arxiv_id="math/1234567v2")
        self.storage.save_paper_metadata(path, [self.metadata, versioned])
        loaded = self.storage.load_paper_input(path, None)
        self.assertEqual(loaded, {self.metadata.arxiv_id: self.metadata})
        record = json.loads(path.read_text().splitlines()[0])
        self.assertEqual(record["authors"], list(self.metadata.authors))
        self.assertEqual(record["publication_date"], "2026-01-02T00:00:00+00:00")

    def test_ids_preserve_requested_versions_and_order(self):
        path = self.root / "ids.txt"
        path.write_text("b\na\nb\n\n1234.56789v2\n", encoding="utf-8")
        self.assertEqual(
            list(self.storage.load_paper_input(None, path)), ["b", "a", "1234.56789v2"]
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.storage.load_paper_input(path, path)

    def test_invalid_metadata_reports_line_number(self):
        path = self.root / "papers.jsonl"
        path.write_text('\n{"authors": "invalid"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            self.storage.load_paper_input(path, None)

    def test_failed_metadata_export_preserves_previous_selection(self):
        path = self.root / "papers.jsonl"
        self.storage.save_paper_metadata(path, [self.metadata])
        original = path.read_bytes()

        def interrupted():
            yield self.metadata
            raise RuntimeError("interrupted")

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.storage.save_paper_metadata(path, interrupted())
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_source_round_trip_preserves_requested_id_and_proof_settings(self):
        requested_id = "math/1234567"
        paper = replace(
            self.paper, metadata=replace(self.metadata, arxiv_id=requested_id + "v2")
        )
        self.storage.save_source_paper(
            self.root, paper, requested_id=requested_id, include_proofs=True
        )
        path = self.storage.source_filepath(self.root, requested_id)
        self.assertEqual(path.name, "math%2F1234567.json.gz")
        with gzip.open(path, "rt") as handle:
            record = json.load(handle)
        self.assertEqual(record["requested_arxiv_id"], requested_id)
        self.assertEqual(self.storage.load_source_paper(self.root, requested_id, True), paper)
        with self.assertRaisesRegex(ValueError, "include-proofs differs"):
            self.storage.load_source_paper(self.root, requested_id, False)

    def test_clusters_and_label_checkpoint_round_trip(self):
        records = [ClusterRecord(1, self.metadata.arxiv_id, self.metadata.title, 0.75)]
        clusters = {1: ClusterMetadata("", 1, -1)}
        self.storage.save_clusters(self.root, records, clusters)
        self.assertEqual(self.storage.load_clusters(self.root), (records, clusters))
        path = self.root / "clusters.csv.gz"
        labelled = {1: ClusterMetadata("Combinatorics", 1, -1)}
        self.storage.save_cluster_metadata(path, labelled, checkpoint=True)
        self.assertEqual(self.storage.load_clusters(self.root), (records, labelled))
        self.assertEqual(
            self.storage.load_memberships(self.root / "memberships.csv.gz")[0]["arxiv_id"],
            self.metadata.arxiv_id,
        )

    def test_failed_label_checkpoint_preserves_existing_progress(self):
        path = self.root / "clusters.csv.gz"
        self.storage.save_cluster_metadata(path, {1: ClusterMetadata("old", 1, -1)})
        original = path.read_bytes()
        with self.assertRaises(AttributeError):
            self.storage.save_cluster_metadata(path, {1: None}, checkpoint=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_coordinates_and_centroids_keep_csv_formats(self):
        path = self.root / "coordinates.gz"
        self.storage.save_coordinates(path, [("a", self.metadata.title)], iter([(1.0, 2.5)]))
        self.assertEqual(self.storage.load_coordinates(path), [{
            "arxiv_id": "a", "title": self.metadata.title, "umap_x": "1.0", "umap_y": "2.5"
        }])
        centroids = self.root / "centroids.gz"
        self.storage.save_centroids(centroids, [(1, (1.0, 2.5))])
        with gzip.open(centroids, "rt") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(json.loads(row["centroid"]), [1.0, 2.5])

    def test_coordinate_count_and_csv_headers_are_validated(self):
        path = self.root / "coordinates.gz"
        with self.assertRaises(ValueError):
            self.storage.save_coordinates(path, [("a", "A")], [])
        with gzip.open(path, "wt") as handle:
            handle.write("arxiv_id,title\na,A\n")
        with self.assertRaisesRegex(ValueError, "umap_x"):
            self.storage.load_coordinates(path)

    def test_manifest_and_required_batch_files(self):
        with self.assertRaisesRegex(ValueError, "Missing manifest"):
            self.storage.read_manifest(self.root)
        manifest = {"arxiv_ids": ["a"], "status": "processing"}
        self.storage.write_manifest(self.root, manifest)
        self.assertEqual(self.storage.read_manifest(self.root), manifest)
        with self.assertRaisesRegex(ValueError, "embeddings.pt.gz"):
            self.storage.validate_embedding_batch_files(self.root)
        for name in ("embeddings.pt.gz", "techniques.pt.gz", "metadata.csv.gz", "techniques.csv.gz"):
            (self.root / name).touch()
        self.storage.validate_embedding_batch_files(self.root)

    def test_batch_folders_must_match(self):
        vectors, metadata = self.root / "vectors", self.root / "metadata"
        (vectors / "batch_0001").mkdir(parents=True)
        (metadata / "batch_0002").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "matching batches"):
            list(self.storage.iter_embedding_rows(vectors, metadata))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_embedding_round_trip_with_and_without_techniques(self):
        for with_techniques in (False, True):
            with self.subTest(with_techniques=with_techniques):
                vector = np.array([0.25, 0.75], dtype=np.float32)
                techniques = [Technique(self.metadata.arxiv_id, "A technique", vector, 0.9)]
                paper = EmbeddedPaper(
                    metadata=self.metadata, sections=(), embedding_dim=2,
                    model_name="test", arxiv_id=self.metadata.arxiv_id,
                    area_vector=vector, techniques=techniques if with_techniques else [],
                )
                batch = self.root / str(with_techniques)
                self.storage.save_embedding_batch(batch, [paper])
                self.storage.validate_embedding_batch_files(batch)
                restored = self.storage.load_embedding_batch(batch)[0]
                self.assertEqual(restored.metadata, paper.metadata)
                self.assertEqual(restored.model_name, paper.model_name)
                np.testing.assert_array_equal(restored.area_vector, vector)
                self.assertEqual(len(restored.techniques), len(paper.techniques))
                if with_techniques:
                    self.assertEqual(restored.techniques[0].text, "A technique")
                    np.testing.assert_array_equal(restored.techniques[0].embedding, vector)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_embedding_reader_rejects_misaligned_metadata(self):
        import torch

        vectors, metadata = self.root / "vectors.pt.gz", self.root / "metadata.csv.gz"
        with gzip.open(vectors, "wb") as handle:
            torch.save(torch.zeros((2, 2)), handle)
        with gzip.open(metadata, "wt") as handle:
            handle.write("arxiv_id,title\na,A\n")
        with self.assertRaisesRegex(ValueError, "fewer rows"):
            list(self.storage.iter_embedding_rows(vectors, metadata))


if __name__ == "__main__":
    unittest.main()
