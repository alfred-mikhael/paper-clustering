"""Workflow checks for LocalStorage integration without models or a database."""

from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from paper_clustering import pipeline
from paper_clustering.data_models import ClusterMetadata, ClusterRecord, Paper, PaperMetadata
from paper_clustering.local_storage import LocalStorage


class PipelineStorageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.storage = LocalStorage()
        storage_patch = patch.object(pipeline, "local_storage", self.storage)
        storage_patch.start()
        self.addCleanup(storage_patch.stop)
        self.metadata = PaperMetadata(
            authors=("Author",), publication_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            arxiv_id="2601.00001", title="Title", abstract="Abstract",
            primary_category="math.CO", url="https://arxiv.org/abs/2601.00001", doi="",
        )

    def test_selection_and_download_use_compatible_storage_formats(self):
        metadata_path = self.root / "papers.jsonl"
        with patch.object(pipeline, "read_metadata_archive", return_value=[self.metadata]):
            self.assertIsNone(pipeline.get_relevant(1, metadata_path))
        paper = Paper(self.metadata, ())
        sources = self.root / "sources"
        with patch(
            "paper_clustering.extract.extract_text.get_paper", return_value=paper
        ) as fetch:
            self.assertIsNone(pipeline.download_papers(metadata_path, output_path=sources))
            self.assertEqual(
                self.storage.load_source_paper(sources, self.metadata.arxiv_id, False), paper
            )
            self.assertIsNone(pipeline.download_papers(
                metadata_path, output_path=sources, resume=True
            ))
            fetch.assert_called_once()

    def test_file_and_database_embedding_inputs_are_separate(self):
        row = {"arxiv_id": "a", "title": "A", "area_embedding": [1, 2]}
        with patch.object(self.storage, "iter_embedding_rows", return_value=iter([(row, [1, 2])])) as read:
            with patch.object(pipeline, "_database_client") as connect:
                self.assertEqual(
                    list(pipeline._embedding_rows(self.root, self.root)), [(row, [1, 2])]
                )
                connect.assert_not_called()
            read.assert_called_once_with(self.root, self.root)
        client = Mock()
        client.select.return_value = [row]
        with patch.object(pipeline, "_database_client", return_value=nullcontext(client)):
            with patch.object(self.storage, "iter_embedding_rows") as read:
                self.assertEqual(list(pipeline._embedding_rows(None, None)), [(row, [1, 2])])
                read.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Supply both"):
            list(pipeline._embedding_rows(self.root, None))

    def make_batch(self, number, paper_id, status, settings):
        path = self.root / f"batch_{number:04d}"
        path.mkdir()
        self.storage.write_manifest(path, {
            **settings, "arxiv_ids": [paper_id], "status": status,
        })
        if status != "processing":
            for name in (
                "embeddings.pt.gz", "techniques.pt.gz", "metadata.csv.gz", "techniques.csv.gz"
            ):
                (path / name).touch()
        return path

    def test_resume_skips_completed_batches_and_recomputes_interrupted_batch(self):
        settings = {"batch_size": 1, "upload": False}
        self.make_batch(1, "a", "completed", settings)
        self.make_batch(2, "b", "processing", settings)
        self.assertEqual(
            pipeline._resume_embedding_offset(["a", "b"], self.root, settings, True), 1
        )
        with self.assertRaisesRegex(ValueError, "Input paper IDs differ"):
            pipeline._resume_embedding_offset(["changed", "b"], self.root, settings, True)

    def test_failed_upload_replay_does_not_mark_batch_complete(self):
        settings = {"batch_size": 1, "upload": True}
        batch = self.make_batch(1, "a", "uploading", settings)
        papers = [SimpleNamespace(arxiv_id="a")]
        with patch.object(self.storage, "load_embedding_batch", return_value=papers):
            with patch.object(
                pipeline, "_upload_embedding_batch", side_effect=RuntimeError("unavailable")
            ) as upload:
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    pipeline._resume_embedding_offset(["a"], self.root, settings, True)
                self.assertEqual(self.storage.read_manifest(batch)["status"], "uploading")
                upload.side_effect = None
                self.assertEqual(
                    pipeline._resume_embedding_offset(["a"], self.root, settings, True), 1
                )
                self.assertEqual(self.storage.read_manifest(batch)["status"], "completed")

    def test_upload_replay_requires_exact_unambiguous_id_coverage(self):
        cases = (
            (["a", "b"], ["a"], False),
            (["a"], [], False),
            (["a", "b"], ["a", "a"], False),
            (["a", "b"], ["av1", "av2"], False),
            (["a", "av1"], ["a", "av1"], False),
            (["av1"], ["av2"], False),
            (["a"], ["other"], False),
            (["a", "b"], ["bv2", "av1"], True),
            (["av1"], ["av1"], True),
        )
        for index, (expected, saved, valid) in enumerate(cases):
            with self.subTest(expected=expected, saved=saved):
                root = self.root / str(index)
                batch = root / "batch_0001"
                batch.mkdir(parents=True)
                settings = {"batch_size": len(expected), "upload": True}
                self.storage.write_manifest(batch, {
                    **settings, "arxiv_ids": expected, "status": "uploading",
                })
                with patch.object(self.storage, "validate_embedding_batch_files"), patch.object(
                    self.storage, "load_embedding_batch",
                    return_value=[SimpleNamespace(arxiv_id=item) for item in saved],
                ), patch.object(pipeline, "_upload_embedding_batch") as upload:
                    if valid:
                        self.assertEqual(
                            pipeline._resume_embedding_offset(expected, root, settings, True),
                            len(expected),
                        )
                        upload.assert_called_once()
                        self.assertEqual(self.storage.read_manifest(batch)["status"], "completed")
                    else:
                        with self.assertRaisesRegex(ValueError, "Saved paper IDs"):
                            pipeline._resume_embedding_offset(expected, root, settings, True)
                        upload.assert_not_called()
                        self.assertEqual(self.storage.read_manifest(batch)["status"], "uploading")

    def test_label_checkpoint_survives_failure_and_is_reused(self):
        module = ModuleType("paper_clustering.cluster")
        module.validate_clustering_options = Mock()
        module.validate_labelling_options = Mock()
        records = [ClusterRecord(1, self.metadata.arxiv_id, "Title", 1.0)]
        clusters = {1: ClusterMetadata("", 1, -1)}
        labelled = {1: ClusterMetadata("Saved label", 1, -1)}
        module.generate_clusters = Mock(return_value=(records, clusters))

        def interrupted_labeling(records, clusters, papers, **kwargs):
            kwargs["on_label"](labelled)
            raise RuntimeError("interrupted")

        module.label_clusters = Mock(side_effect=interrupted_labeling)
        client = Mock()
        client.transaction.side_effect = lambda: nullcontext()
        client.insert.return_value = None
        record = {**asdict(self.metadata), "embedding_dim": 2, "model_name": "test"}
        with patch.dict(sys.modules, {module.__name__: module}), patch.object(
            pipeline, "_embedding_rows", return_value=[(record, [1.0, 0.0])]
        ), patch.object(pipeline, "_database_client", return_value=nullcontext(client)):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                pipeline.cluster_papers(self.root)
            self.assertEqual(self.storage.load_clusters(self.root), (records, labelled))
            client.insert.assert_not_called()
            module.label_clusters.side_effect = lambda records, clusters, papers, **kw: clusters
            self.assertIsNone(pipeline.cluster_papers(self.root))
            module.generate_clusters.assert_called_once()
            self.assertEqual(module.label_clusters.call_args.args[1], labelled)
            self.assertEqual(client.insert.call_count, 2)
            self.assertEqual(self.storage.read_manifest(self.root)["status"], "completed")
            pipeline.cluster_papers(self.root)
            self.assertEqual(client.insert.call_count, 2)
            self.assertEqual(module.label_clusters.call_count, 2)
            for changes in ({"min_cluster_size": 7}, {"model": "different"}, {"seed": 10}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "differs"):
                    pipeline.cluster_papers(self.root, **changes)
            for changed_record, vector in (
                ({**record, "abstract": "changed"}, [1.0, 0.0]),
                (record, [0.0, 1.0]),
            ):
                with patch.object(pipeline, "_embedding_rows", return_value=[(changed_record, vector)]):
                    with self.assertRaisesRegex(ValueError, "input_fingerprint differs"):
                        pipeline.cluster_papers(self.root)

    def test_interrupted_clustering_recomputes_partial_exports(self):
        module = ModuleType("paper_clustering.cluster")
        module.validate_clustering_options = Mock()
        module.validate_labelling_options = Mock()
        module.generate_clusters = Mock(return_value=([], {}))
        module.label_clusters = Mock()
        with patch.dict(sys.modules, {module.__name__: module}), patch.object(
            pipeline, "_embedding_rows", return_value=[]
        ):
            with patch.object(self.storage, "save_clusters", side_effect=RuntimeError("disk failed")):
                with self.assertRaisesRegex(RuntimeError, "disk failed"):
                    pipeline.cluster_papers(self.root)
            self.assertEqual(self.storage.read_manifest(self.root)["status"], "clustering")
            (self.root / "memberships.csv.gz").write_bytes(b"incomplete")
            pipeline.cluster_papers(self.root)
            self.assertEqual(module.generate_clusters.call_count, 2)
            self.assertEqual(self.storage.read_manifest(self.root)["status"], "completed")
            module.label_clusters.assert_not_called()

    def test_legacy_cluster_files_require_new_output_directory(self):
        module = ModuleType("paper_clustering.cluster")
        module.validate_clustering_options = Mock()
        module.validate_labelling_options = Mock()
        module.generate_clusters = Mock()
        module.label_clusters = Mock()
        self.storage.save_clusters(self.root, [], {})
        with patch.dict(sys.modules, {module.__name__: module}), patch.object(
            pipeline, "_embedding_rows", return_value=[]
        ), self.assertRaisesRegex(ValueError, "no manifest"):
            pipeline.cluster_papers(self.root)
        module.generate_clusters.assert_not_called()

    def test_cli_keeps_embedding_defaults(self):
        with patch.object(pipeline, "embed_papers", return_value=None) as embed:
            self.assertEqual(pipeline.main([
                "embed", "--input-filepath", "papers.jsonl", "--output-path", "out"
            ]), 0)
        self.assertEqual(embed.call_args.kwargs["pooling"], "max")
        self.assertEqual(embed.call_args.kwargs["technique_k"], 3.0)


if __name__ == "__main__":
    unittest.main()
