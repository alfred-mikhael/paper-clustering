"""Embedding upload conversion and transaction boundaries."""

from contextlib import nullcontext
from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from paper_clustering import pipeline
from paper_clustering.data_models import EmbeddedPaper, PaperMetadata, Technique
from paper_clustering.interfaces import DatabaseClient


class PipelineUploadTests(unittest.TestCase):
    def setUp(self):
        for target in ("paper_clustering.utils.sleep", "paper_clustering.utils.logger"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        metadata = PaperMetadata(
            authors=("Author",), publication_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            arxiv_id="2601.00001", title="Title", abstract="Abstract",
            primary_category="math.CO", url="https://arxiv.org/abs/2601.00001", doi="",
        )
        self.paper = EmbeddedPaper(
            metadata=metadata, sections=(), embedding_dim=2, model_name="test",
            arxiv_id=metadata.arxiv_id, area_vector=np.array([1.0, 0.0]),
            techniques=[Technique(metadata.arxiv_id, "Passage", np.array([0.0, 1.0]), 0.8)],
        )
        self.client = Mock(spec=DatabaseClient)
        self.client.insert.return_value = None
        self.transaction = MagicMock()
        self.transaction.__exit__.return_value = False
        self.client.transaction.return_value = self.transaction

    def test_empty_batch_does_not_open_connection(self):
        with patch.object(pipeline, "_database_client") as connect:
            self.assertIsNone(pipeline._upload_embedding_batch([]))
        connect.assert_not_called()

    def test_converts_records_and_groups_inserts_in_one_transaction(self):
        def insert(*, table, records):
            self.transaction.__enter__.assert_called_once()
            self.transaction.__exit__.assert_not_called()

        self.client.insert.side_effect = insert
        with patch.object(pipeline, "_database_client", return_value=nullcontext(self.client)):
            self.assertIsNone(pipeline._upload_embedding_batch([self.paper]))
        self.client.transaction.assert_called_once()
        self.transaction.__exit__.assert_called_once_with(None, None, None)
        papers_call, techniques_call = self.client.insert.call_args_list
        self.assertEqual(papers_call.kwargs["table"], "papers")
        record = papers_call.kwargs["records"][0]
        self.assertEqual(record["authors"], ["Author"])
        self.assertEqual(record["publication_date"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(record["arxiv_id"], self.paper.arxiv_id)
        self.assertIs(record["area_embedding"], self.paper.area_vector)
        self.assertEqual(techniques_call.kwargs["table"], "techniques")
        technique = techniques_call.kwargs["records"][0]
        self.assertEqual(technique["passage"], "Passage")
        self.assertEqual(technique["score"], 0.8)
        self.assertIs(technique["embedding"], self.paper.techniques[0].embedding)

    def test_insert_failure_reaches_transaction_exit_and_caller(self):
        for failing_insert in (1, 2):
            with self.subTest(failing_insert=failing_insert):
                error = RuntimeError("database failure")
                self.client.reset_mock()
                self.transaction.reset_mock()
                self.client.insert.side_effect = (
                    [None] * (failing_insert - 1) + [error]
                ) * 4
                with patch.object(
                    pipeline, "_database_client", return_value=nullcontext(self.client)
                ), self.assertRaises(RuntimeError) as caught:
                    pipeline._upload_embedding_batch([self.paper])
                self.assertIs(caught.exception, error)
                self.assertEqual(self.client.insert.call_count, failing_insert * 4)
                self.assertEqual(self.client.transaction.call_count, 4)
                self.assertIs(self.transaction.__exit__.call_args.args[1], error)

    def test_failed_techniques_upload_retries_both_tables_on_new_connection(self):
        error = RuntimeError("connection lost")
        self.client.insert.side_effect = [None, error]
        replacement = Mock(spec=DatabaseClient)
        replacement.transaction.return_value = nullcontext()
        with patch.object(pipeline, "_database_client", side_effect=[
            nullcontext(self.client), nullcontext(replacement)
        ]) as connect:
            pipeline._upload_embedding_batch([self.paper])
        self.assertEqual(connect.call_count, 2)
        self.assertIs(self.transaction.__exit__.call_args.args[1], error)
        self.assertEqual(
            [call.kwargs["table"] for call in replacement.insert.call_args_list],
            ["papers", "techniques"],
        )

    def test_commit_failure_retries_the_whole_upsert(self):
        self.transaction.__exit__.side_effect = RuntimeError("commit result unknown")
        replacement = Mock(spec=DatabaseClient)
        replacement.transaction.return_value = nullcontext()
        with patch.object(pipeline, "_database_client", side_effect=[
            nullcontext(self.client), nullcontext(replacement)
        ]):
            pipeline._upload_embedding_batch([self.paper])
        self.assertEqual(replacement.insert.call_count, 2)

    def test_argument_errors_do_not_retry_uploads(self):
        for error in (ValueError("invalid row"), TypeError("invalid type")):
            self.client.insert.side_effect = error
            with self.subTest(error=type(error)), patch.object(
                pipeline, "_database_client", return_value=nullcontext(self.client)
            ) as connect, self.assertRaises(type(error)):
                pipeline._upload_embedding_batch([self.paper])
            connect.assert_called_once()

    def test_connection_failure_is_retried(self):
        with patch.object(pipeline, "_database_client", side_effect=[
            RuntimeError("connection refused"), nullcontext(self.client)
        ]) as connect:
            pipeline._upload_embedding_batch([self.paper])
        self.assertEqual(connect.call_count, 2)
        self.client.transaction.assert_called_once()


class PipelineReadRetryTests(unittest.TestCase):
    def test_read_reconnects_before_yielding_rows(self):
        failed = Mock(spec=DatabaseClient)
        failed.select.side_effect = RuntimeError("connection lost during fetch")
        replacement = Mock(spec=DatabaseClient)
        record = {"arxiv_id": "a", "title": "Title", "area_embedding": [1.0]}
        replacement.select.return_value = [record]
        with patch.object(pipeline, "_database_client", side_effect=[
            nullcontext(failed), nullcontext(replacement)
        ]) as connect, patch("paper_clustering.utils.sleep"), patch("paper_clustering.utils.logger"):
            rows = list(pipeline._embedding_rows(None, None, full_metadata=True))
        self.assertEqual(connect.call_count, 2)
        replacement.select.assert_called_once_with(table="papers", cols=["*"])
        self.assertEqual(rows, [(record, record["area_embedding"])])


if __name__ == "__main__":
    unittest.main()
