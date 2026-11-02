"""Database retrieval retries must release connections before reacquiring them."""

from contextlib import contextmanager
import importlib
import importlib.util
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class ApiRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in ("psycopg", "psycopg_pool", "pgvector", "fastapi")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise unittest.SkipTest("Missing API dependencies: " + ", ".join(missing))
        # These tests exercise database boundaries, without loading a GPU model.
        reranker = ModuleType("paper_clustering.reranker")
        reranker.SLMReranker = Mock()
        with patch.dict(sys.modules, {reranker.__name__: reranker}):
            cls.api = importlib.import_module("paper_clustering.api")

    def setUp(self):
        for target in ("paper_clustering.utils.sleep", "paper_clustering.utils.logger"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_failed_retrieval_releases_connection_before_retry(self):
        events = []
        clients = [Mock(), Mock()]
        request = Mock()

        @contextmanager
        def connect(request):
            index = len(events) // 2
            events.append(("enter", index))
            try:
                yield clients[index]
            finally:
                events.append(("exit", index))

        with patch.object(self.api, "get_database_client", side_effect=connect), patch.object(
            self.api, "retrieve_candidates", side_effect=[RuntimeError("read failed"), []]
        ) as retrieve:
            self.assertEqual(self.api._retrieve_candidates(request, "a", 10), [])
        self.assertEqual(events, [("enter", 0), ("exit", 0), ("enter", 1), ("exit", 1)])
        self.assertEqual([call.args[1] for call in retrieve.call_args_list], clients)

    def test_unavailable_database_uses_bounded_retries(self):
        error = self.api.HTTPException(status_code=503, detail="Database unavailable")
        with patch.object(self.api, "get_database_client", side_effect=error) as connect:
            with self.assertRaises(self.api.HTTPException) as caught:
                self.api._retrieve_candidates(Mock(), "a", 10)
        self.assertIs(caught.exception, error)
        self.assertEqual(connect.call_count, 4)

    def test_invalid_arguments_are_not_retried(self):
        for error in (TypeError("invalid type"), ValueError("invalid argument")):
            with self.subTest(error=type(error)), patch.object(
                self.api, "get_database_client", side_effect=error
            ) as connect, self.assertRaises(type(error)):
                self.api._retrieve_candidates(Mock(), "a", 10)
            connect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
