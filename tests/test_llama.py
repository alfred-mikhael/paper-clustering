"""HTTP payloads and error translation using a mocked session."""

import importlib.util
import os
import unittest
from unittest.mock import patch

import requests


@unittest.skipIf(importlib.util.find_spec("dotenv") is None, "python-dotenv is not installed")
class LlamaConnectorTests(unittest.TestCase):
    def setUp(self):
        from paper_clustering.llama import LlamaConnector
        self.connector_type = LlamaConnector
        for target in ("paper_clustering.llama.load_dotenv", "paper_clustering.llama.requests.Session"):
            patcher = patch(target)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if target.endswith("Session"):
                self.session = mocked.return_value

    def test_payload_timeout_grammar_and_session_reuse(self):
        connector = self.connector_type("http://server:8080/", request_timeout=15)
        self.session.post.return_value.json.return_value = {"choices": []}
        messages = [{"role": "user", "content": "Prompt"}]
        self.assertEqual(connector.complete(messages, grammar="grammar", max_tokens=2), {"choices": []})
        self.session.post.assert_called_once_with(
            "http://server:8080/v1/chat/completions",
            json={"messages": messages, "grammar": "grammar", "max_tokens": 2}, timeout=15,
        )
        connector.complete(messages)
        self.assertNotIn("grammar", self.session.post.call_args.kwargs["json"])

    def test_environment_endpoint_and_invalid_timeout(self):
        with patch.dict(os.environ, {"LLAMA_CPP_SERVER_URL": "http://environment/"}):
            self.assertEqual(self.connector_type().server_url, "http://environment")
            self.assertEqual(self.connector_type("http://explicit").server_url, "http://explicit")
        for timeout in (0, -1):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.connector_type(request_timeout=timeout)

    def test_network_error_preserves_cause(self):
        connector = self.connector_type("http://server")
        error = requests.Timeout("timed out")
        self.session.post.side_effect = error
        with self.assertRaisesRegex(RuntimeError, "http://server") as caught:
            connector.complete([])
        self.assertIs(caught.exception.__cause__, error)

    def test_invalid_server_json_preserves_cause_and_bounds_error_body(self):
        connector = self.connector_type("http://server")
        response = self.session.post.return_value
        response.text = "x" * 500 + "END-OF-BODY"
        error = ValueError("invalid JSON")
        response.json.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            connector.complete([])
        self.assertIs(caught.exception.__cause__, error)
        self.assertNotIn("END-OF-BODY", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
