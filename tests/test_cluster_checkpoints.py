"""Cluster upload progress, commit retries, and checkpoint recovery."""

from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import MagicMock, Mock, patch

from paper_clustering import pipeline
from paper_clustering.data_models import ClusterMetadata, ClusterRecord, PaperMetadata
from paper_clustering.local_storage import LocalStorage


class ClusterCheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.storage = LocalStorage()
        metadata = PaperMetadata(
            authors=("Author",), publication_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            arxiv_id="a", title="A", abstract="Abstract", primary_category="math.CO", url="", doi="",
        )
        self.records = [
            ClusterRecord(1, "a", "A", 1.0), ClusterRecord(1, "b", "B", 1.0),
            ClusterRecord(2, "a", "A", 1.0),
        ]
        clusters = {1: ClusterMetadata("", 2, -1), 2: ClusterMetadata("", 1, 1)}
        labels = {1: ClusterMetadata("Root", 2, -1), 2: ClusterMetadata("Child", 1, 1)}
        self.module = ModuleType("paper_clustering.cluster")
        self.module.validate_clustering_options = Mock()
        self.module.validate_labelling_options = Mock()
        self.module.generate_clusters = Mock(return_value=(self.records, clusters))
        self.module.label_clusters = Mock(return_value=labels)
        self.upload = Mock()
        for patcher in (
            patch.dict(sys.modules, {self.module.__name__: self.module}),
            patch.object(pipeline, "local_storage", self.storage),
            patch.object(pipeline, "CLUSTER_UPLOAD_BATCH_SIZE", 1),
            patch.object(pipeline, "_embedding_rows", return_value=[(asdict(metadata), [1.0, 0.0])]),
            patch.object(pipeline, "_upload_cluster_batch", self.upload),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_resume_allows_changing_server_url_including_legacy_manifest(self):
        labels = self.module.label_clusters.return_value
        self.module.label_clusters.side_effect = RuntimeError("server unavailable")
        with self.assertRaisesRegex(RuntimeError, "server unavailable"):
            pipeline.cluster_papers(self.root, server_url="http://old-server:8080")
        manifest = self.storage.read_manifest(self.root)
        self.assertNotIn("server_url", manifest)
        # Earlier checkpoints included the endpoint; it must no longer constrain resume.
        manifest["server_url"] = "http://old-server:8080"
        self.storage.write_manifest(self.root, manifest)
        self.module.label_clusters.side_effect = None
        self.module.label_clusters.return_value = labels
        pipeline.cluster_papers(self.root, server_url="http://new-server:8080")
        self.module.generate_clusters.assert_called_once()
        self.assertEqual(
            self.module.label_clusters.call_args.kwargs["server_url"], "http://new-server:8080"
        )
        self.assertEqual(self.storage.read_manifest(self.root)["status"], "completed")

    def test_resume_skips_committed_batches_and_completed_labelling(self):
        self.upload.side_effect = [None, None, None, RuntimeError("database unavailable")]
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            pipeline.cluster_papers(self.root)
        manifest = self.storage.read_manifest(self.root)
        self.assertEqual(manifest["status"], "uploading")
        self.assertEqual(manifest["uploaded_rows"], {"cluster_metadata": 2, "clusters": 1})
        self.upload.reset_mock()
        self.upload.side_effect = None
        pipeline.cluster_papers(self.root)
        self.assertEqual(
            [call.args for call in self.upload.call_args_list],
            [("clusters", [asdict(record)]) for record in self.records[1:]],
        )
        self.module.generate_clusters.assert_called_once()
        self.module.label_clusters.assert_called_once()
        self.assertEqual(self.storage.read_manifest(self.root)["status"], "completed")

    def test_commit_without_saved_progress_replays_that_batch(self):
        write = self.storage.write_manifest

        def fail_checkpoint(path, manifest):
            if manifest["uploaded_rows"]["cluster_metadata"] == 1:
                raise OSError("checkpoint failed")
            write(path, manifest)

        with patch.object(self.storage, "write_manifest", side_effect=fail_checkpoint):
            with self.assertRaisesRegex(OSError, "checkpoint failed"):
                pipeline.cluster_papers(self.root)
        first_upload = self.upload.call_args
        self.assertEqual(self.storage.read_manifest(self.root)["uploaded_rows"]["cluster_metadata"], 0)
        self.upload.reset_mock()
        pipeline.cluster_papers(self.root)
        self.assertEqual(self.upload.call_args_list[0], first_upload)
        self.assertEqual(self.upload.call_count, 5)
        self.module.label_clusters.assert_called_once()

    def test_invalid_progress_is_rejected_before_upload(self):
        self.upload.side_effect = RuntimeError("unavailable")
        with self.assertRaises(RuntimeError):
            pipeline.cluster_papers(self.root)
        manifest = self.storage.read_manifest(self.root)
        manifest["uploaded_rows"]["clusters"] = 100
        self.storage.write_manifest(self.root, manifest)
        self.upload.reset_mock()
        with self.assertRaisesRegex(ValueError, "Invalid cluster upload progress"):
            pipeline.cluster_papers(self.root)
        self.upload.assert_not_called()


class ClusterUploadRetryTests(unittest.TestCase):
    def test_commit_failure_retries_on_new_connection(self):
        first, second = Mock(), Mock()
        transaction = MagicMock()
        error = RuntimeError("commit result unknown")
        transaction.__exit__.side_effect = error
        first.transaction.return_value = transaction
        second.transaction.return_value = nullcontext()
        records = [{"id": 1, "label": "Topic"}]
        with patch.object(pipeline, "_database_client", side_effect=[
            nullcontext(first), nullcontext(second)
        ]) as connect, patch("paper_clustering.utils.sleep"), patch("paper_clustering.utils.logger"):
            pipeline._upload_cluster_batch("cluster_metadata", records)
        self.assertEqual(connect.call_count, 2)
        first.insert.assert_called_once_with(table="cluster_metadata", records=records)
        second.insert.assert_called_once_with(table="cluster_metadata", records=records)


if __name__ == "__main__":
    unittest.main()
