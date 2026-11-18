"""Provider adaptation and notebook switching without paid API calls."""

import ast
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter
import unittest
from unittest.mock import Mock, patch

import requests


@unittest.skipIf(importlib.util.find_spec("dotenv") is None, "python-dotenv is not installed")
class ConnectorTests(unittest.TestCase):
    def setUp(self):
        from paper_clustering.connector import Connector, LlamaConnector, PerplexityConnector
        self.interface = Connector
        self.llama = LlamaConnector
        self.perplexity = PerplexityConnector
        patcher = patch("paper_clustering.connector.load_dotenv")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.session = Mock()

    def test_perplexity_authentication_payload_and_close(self):
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-key",
                                     "PERPLEXITY_MODEL": "example/model"}):
            connector = self.perplexity(session=self.session, request_timeout=12)
        self.assertIsInstance(connector, self.interface)
        self.session.headers.update.assert_called_once_with({"Authorization": "Bearer test-key"})
        response = {"choices": [{"message": {"content": "answer"}}]}
        self.session.post.return_value.json.return_value = response
        messages = [{"role": "user", "content": "question"}]
        actual = connector.complete(messages, max_tokens=10, cache_prompt=True,
                                    top_k=64, chat_template_kwargs={"enable_thinking": False},
                                    response_format={"type": "json_object"})
        self.assertEqual(actual, response)
        self.session.post.assert_called_once_with(
            "https://api.perplexity.ai/orchestrator/v1/chat/completions",
            json={"model": "example/model", "messages": messages, "max_tokens": 10,
                  "response_format": {"type": "json_object"}}, timeout=12)
        self.session.post.reset_mock()
        self.assertIsNone(connector.count_prompt_tokens(messages))
        self.session.post.assert_not_called()
        connector.close()
        self.session.close.assert_called_once()

    def test_missing_key_and_unsupported_modes_fail_before_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PERPLEXITY_API_KEY"):
                self.perplexity(session=self.session)
        connector = self.perplexity(api_key="test-key", session=self.session)
        for options in ({"grammar": "root ::= 'A'"}, {"stream": True}):
            with self.assertRaises(ValueError):
                connector.complete([], **options)
        self.session.post.assert_not_called()
        with self.assertRaises(ValueError):
            self.perplexity(api_key="test-key", request_timeout=0)

    def test_explicit_settings_override_environment_and_errors_redact_key(self):
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "env-key", "PERPLEXITY_MODEL": "env-model"}):
            connector = self.perplexity(api_key="explicit-key", model="explicit-model",
                                        session=self.session)
        self.assertEqual(connector.model, "explicit-model")
        self.session.headers.update.assert_called_once_with({"Authorization": "Bearer explicit-key"})
        self.session.post.return_value.text = "Error: explicit-key"
        self.session.post.return_value.raise_for_status.side_effect = requests.HTTPError("401")
        with self.assertRaises(RuntimeError) as caught:
            connector.complete([])
        self.assertNotIn("explicit-key", str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))

    def test_llama_exact_preflight(self):
        connector = self.llama("http://local", session=self.session)
        self.session.post.return_value.json.side_effect = [
            {"prompt": "rendered"}, {"tokens": [1, 2, 3]},
        ]
        self.assertEqual(connector.count_prompt_tokens([]), 3)
        calls = self.session.post.call_args_list
        self.assertEqual(calls[0].args, ("http://local/apply-template",))
        self.assertEqual(calls[1].kwargs["json"],
                         {"content": "rendered", "add_special": True, "parse_special": True})

    def notebook_functions(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/extract_reductions.ipynb"
        notebook = json.loads(path.read_text())
        names = {"create_connector", "count_prompt_tokens", "extract_connections",
                 "validate_response", "validate_connection", "validate_evidence"}
        functions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                functions.extend(node for node in ast.parse("".join(cell["source"])).body
                                 if isinstance(node, ast.FunctionDef) and node.name in names)
        namespace = dict(Connector=self.interface, LlamaConnector=self.llama,
                         PerplexityConnector=self.perplexity, json=json, perf_counter=perf_counter,
                         TEACHER_PROMPT="Extract connections", MAX_OUTPUT_TOKENS=100,
                         TOKEN_MARGIN=10, CONTEXT_WINDOW_TOKENS=200, SEED=42,
                         REQUEST_TIMEOUT_SECONDS=15, PERPLEXITY_MODEL="example/model",
                         POSITIVE_CLASSES=("cross_field", "same_field"))
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        return namespace

    def test_notebook_provider_selection(self):
        namespace = self.notebook_functions()
        for name, class_name in (("llama", "LlamaConnector"), ("perplexity", "PerplexityConnector")):
            factory = Mock()
            namespace[class_name] = factory
            namespace["CONNECTOR_PROVIDER"] = name
            self.assertIs(namespace["create_connector"](), factory.return_value)
        namespace["CONNECTOR_PROVIDER"] = "typo"
        with self.assertRaises(ValueError):
            namespace["create_connector"]()

    def test_notebook_perplexity_has_no_local_preflight_or_local_options(self):
        namespace = self.notebook_functions()
        connector = self.perplexity(api_key="test-key", session=self.session)
        response = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"disposition": "no_connection", "connections": []})}}],
            "usage": {"prompt_tokens": 123}}
        self.session.post.return_value.json.return_value = response
        result = namespace["extract_connections"](connector, "Title", [])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["token_preflight"], "unavailable")
        self.assertEqual(result["input_tokens"], 123)
        self.session.post.assert_called_once()
        payload = self.session.post.call_args.kwargs["json"]
        for option in ("seed", "cache_prompt", "chat_template_kwargs"):
            self.assertNotIn(option, payload)
        response["choices"][0]["finish_reason"] = "length"
        result = namespace["extract_connections"](connector, "Title", [])
        self.assertEqual(result["status"], "error")
        self.assertIs(result["raw_response"], response)

    def test_notebook_oversized_llama_input_skips_inference(self):
        namespace = self.notebook_functions()
        connector = self.llama(session=self.session)
        result = namespace["extract_connections"](connector, "Title", [],
                                                    token_counter=lambda *_: 100)
        self.assertEqual(result["status"], "oversized")
        self.assertFalse(result["request_attempted"])
        self.session.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
