"""LLM response errors, with unused model-loading dependencies stubbed if absent."""

from contextlib import nullcontext
import importlib
import importlib.util
import json
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from paper_clustering import utils
from paper_clustering.data_models import ClusterMetadata, ClusterRecord, Paper, Statement


class LlmErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These tests exercise response parsing, never model inference or clustering.
        stubs = {}
        for name, attributes in (
            ("dotenv", {"load_dotenv": Mock()}),
            ("hdbscan", {"HDBSCAN": Mock(side_effect=AssertionError("unexpected clustering"))}),
            ("torch", {"Tensor": object}),
        ):
            if importlib.util.find_spec(name) is None:
                stub = ModuleType(name)
                stub.__dict__.update(attributes)
                stubs[name] = stub
        with patch.dict(sys.modules, stubs):
            cls.context = importlib.import_module("paper_clustering.contextualize")
            cls.cluster = importlib.import_module("paper_clustering.cluster")
            cls.labels = importlib.import_module("paper_clustering.label.slm_labels")

    def setUp(self):
        for name in ("sleep", "logger"):
            patcher = patch.object(utils, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.metadata = SimpleNamespace(arxiv_id="a", title="Title", abstract="Abstract")
        self.paper = Paper(self.metadata, ())
        self.statements = [Statement("Intro", "Intro", "lemma", "Statement")]
        self.records = [ClusterRecord(1, "a", "Title", 1.0)]
        self.clusters = {1: ClusterMetadata("", 1, -1)}
        progress = patch("tqdm.tqdm", return_value=nullcontext(Mock()))
        progress.start()
        self.addCleanup(progress.stop)

    def response(self, content):
        return {"choices": [{"message": {"content": content}}]}

    def test_malformed_message_is_a_runtime_error_for_both_consumers(self):
        responses = ({}, {"choices": []}, self.response(None), self.response(" "))
        for response in responses:
            for function in ("contextualize", "select_relevant_statements"):
                with self.subTest(response=response, function=function):
                    connector = Mock()
                    connector.complete.return_value = response
                    argument = self.statements[0] if function == "contextualize" else self.statements
                    with self.assertRaises(RuntimeError):
                        getattr(self.context, function)(connector, argument, self.paper)

    def test_selection_preserves_parse_cause_and_rejects_invalid_indices(self):
        connector = Mock()
        connector.complete.return_value = self.response("not JSON")
        with self.assertRaises(RuntimeError) as caught:
            self.context.select_relevant_statements(connector, self.statements, self.paper)
        self.assertIsInstance(caught.exception.__cause__, json.JSONDecodeError)
        for content in ("{}", "[]", "[0]", "[2]", "[true]", "[1, 1]"):
            connector.complete.return_value = self.response(content)
            with self.subTest(content=content), self.assertRaises(RuntimeError):
                self.context.select_relevant_statements(connector, self.statements, self.paper)

    def test_valid_selection_and_summary_still_work(self):
        connector = Mock()
        connector.complete.return_value = self.response("[1]")
        self.assertEqual(
            self.context.select_relevant_statements(connector, self.statements, self.paper),
            self.statements,
        )
        connector.complete.return_value = self.response("  A summary.  ")
        self.assertEqual(
            self.context.contextualize(connector, self.statements[0], self.paper), "A summary."
        )
        connector.complete.reset_mock()
        self.assertEqual(self.context.select_relevant_statements(connector, [], self.paper), [])
        connector.complete.assert_not_called()

    def test_runtime_response_failure_retries_request_and_validation(self):
        connector = Mock()
        connector.complete.side_effect = [{}, self.response("[1]")]
        self.assertEqual(
            self.context.select_relevant_statements(connector, self.statements, self.paper),
            self.statements,
        )
        self.assertEqual(connector.complete.call_count, 2)

    def test_contextualize_retries_network_error_then_malformed_response(self):
        connector = Mock()
        connector.complete.side_effect = [RuntimeError("timeout"), {}, self.response("Summary")]
        self.assertEqual(
            self.context.contextualize(connector, self.statements[0], self.paper), "Summary"
        )
        self.assertEqual(connector.complete.call_count, 3)

    def test_llm_argument_errors_do_not_retry(self):
        for error in (TypeError("invalid type"), ValueError("invalid argument")):
            for function, argument in (
                (self.context.contextualize, self.statements[0]),
                (self.context.select_relevant_statements, self.statements),
            ):
                connector = Mock()
                connector.complete.side_effect = error
                with self.subTest(function=function.__name__, error=type(error)), self.assertRaises(type(error)):
                    function(connector, argument, self.paper)
                connector.complete.assert_called_once()

    def test_paragraph_response_validation_retries_only_that_paragraph(self):
        labeler = self.labels.SLMWeakLabelGen.__new__(self.labels.SLMWeakLabelGen)
        labeler.connector = Mock()
        labeler._probabilities_from_response = Mock(side_effect=[RuntimeError("invalid probabilities"), "valid"])
        self.assertEqual(labeler._label_paragraph([]), "valid")
        self.assertEqual(labeler.connector.complete.call_count, 2)

    def test_cluster_invalid_json_retries_and_preserves_last_cause(self):
        generate = Mock(return_value="not JSON")
        with patch.object(self.cluster.logger, "warning"), self.assertRaises(RuntimeError) as caught:
            self.cluster.label_clusters(
                self.records, self.clusters, [self.metadata], generate=generate
            )
        self.assertEqual(generate.call_count, 3)
        self.assertIsInstance(caught.exception.__cause__, json.JSONDecodeError)

    def test_cluster_invalid_labels_retry_then_succeed(self):
        generate = Mock(side_effect=['[]', '{"2": "wrong ID"}', '{"1": "Topic"}'])
        with patch.object(self.cluster.logger, "warning"):
            result = self.cluster.label_clusters(
                self.records, self.clusters, [self.metadata], generate=generate
            )
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(result[1].label, "Topic")

    def test_group_checkpoints_preserve_sampling_after_resume(self):
        clusters = {
            cid: ClusterMetadata("", 3, parent)
            for cid, parent in ((1, -1), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3))
        }
        papers = [
            SimpleNamespace(arxiv_id=f"{cid}-{i}", title=f"Paper {cid}-{i}", abstract="Abstract")
            for cid in (4, 5, 6, 7) for i in range(5)
        ]
        records = [
            ClusterRecord(int(paper.arxiv_id.split("-")[0]), paper.arxiv_id, paper.title, 1.0)
            for paper in papers
        ]

        def generate(prompt):
            keys = json.loads(prompt.split("Required output keys (exactly): ")[1].split("\n")[0])
            return json.dumps({key: f"Topic {key}" for key in keys})

        complete = Mock(side_effect=generate)
        checkpoints = []
        self.cluster.label_clusters(
            records, clusters, papers, random_state=42, num_samples=2,
            generate=complete, on_label=lambda result: checkpoints.append(dict(result)),
        )
        self.assertEqual(len(checkpoints), 4)
        self.assertEqual(set(checkpoints[0]), {6, 7})
        resumed = Mock(side_effect=generate)
        self.cluster.label_clusters(
            records, clusters | checkpoints[0], papers, random_state=42, num_samples=2,
            generate=resumed,
        )
        self.assertEqual(resumed.call_args_list, complete.call_args_list[1:])

    def test_checkpoint_failure_does_not_repeat_llm_request(self):
        generate = Mock(return_value='{"1": "Topic"}')
        save = Mock(side_effect=RuntimeError("disk failed"))
        with self.assertRaisesRegex(RuntimeError, "disk failed"):
            self.cluster.label_clusters(
                self.records, self.clusters, [self.metadata], generate=generate, on_label=save,
            )
        generate.assert_called_once()
        save.assert_called_once()

    def test_unhandled_argument_errors_are_not_retried(self):
        for error in (TypeError("bad argument"), ValueError("bad setting")):
            generate = Mock(side_effect=error)
            with self.subTest(error=type(error)), self.assertRaises(type(error)) as caught:
                self.cluster.label_clusters(
                    self.records, self.clusters, [self.metadata], generate=generate
                )
            self.assertIs(caught.exception, error)
            generate.assert_called_once()

    def test_malformed_http_message_is_runtime_error(self):
        response = Mock()
        response.json.return_value = {}
        with patch.object(self.cluster.requests, "post", return_value=response) as request:
            with patch.object(self.cluster.logger, "warning"), self.assertRaises(RuntimeError) as caught:
                self.cluster.label_clusters(self.records, self.clusters, [self.metadata])
        self.assertEqual(request.call_count, 3)
        self.assertIsInstance(caught.exception.__cause__, KeyError)

    def test_invalid_token_probability_is_runtime_error(self):
        for candidate in ({"token": "A", "prob": "invalid"}, {"token": "A", "logprob": "invalid"}):
            response = {"choices": [{"logprobs": {"content": [{"top_probs": [candidate]}]}}]}
            with self.subTest(candidate=candidate), self.assertRaises(RuntimeError) as caught:
                self.labels.SLMWeakLabelGen._probabilities_from_response(response)
            self.assertIsInstance(caught.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
