"""CLI exit codes and exception-based workflow contracts."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from paper_clustering import pipeline


COMMANDS = (
    ("get_relevant", ["select", "--days-back", "1", "--output-filepath", "out"]),
    ("download_papers", ["download", "--ids-filepath", "ids", "--output-path", "out"]),
    ("embed_papers", ["embed", "--input-filepath", "papers", "--output-path", "out"]),
    ("generate_coordinates", ["generate_coords", "--output-filepath", "out"]),
    ("generate_centroids", ["generate_centroids", "--memberships-filepath", "members",
                            "--coordinates-filepath", "coords", "--output-filepath", "out"]),
    ("cluster_papers", ["cluster", "--output-path", "out"]),
)


class PipelineCliTests(unittest.TestCase):
    def test_all_commands_return_zero_on_successful_none_result(self):
        for function, argv in COMMANDS:
            with self.subTest(command=argv[0]), patch.object(
                pipeline, function, return_value=None
            ) as workflow, patch.object(pipeline.logger, "exception") as log:
                self.assertEqual(pipeline.main(argv), 0)
                workflow.assert_called_once()
                log.assert_not_called()

    def test_all_commands_log_final_exception_once_and_return_one(self):
        for function, argv in COMMANDS:
            for error in (RuntimeError("operation failed"), ValueError("invalid input")):
                with self.subTest(command=argv[0], error=type(error)), patch.object(
                    pipeline, function, side_effect=error
                ), self.assertLogs(pipeline.logger, level="ERROR") as logs:
                    self.assertEqual(pipeline.main(argv), 1)
                self.assertEqual(len(logs.records), 1)
                self.assertIs(logs.records[0].exc_info[1], error)

    def test_interrupts_and_system_exit_propagate(self):
        for error in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(error=type(error)), patch.object(
                pipeline, "embed_papers", side_effect=error
            ), patch.object(pipeline.logger, "exception") as log:
                with self.assertRaises(type(error)) as caught:
                    pipeline.main(COMMANDS[2][1])
                self.assertIs(caught.exception, error)
                log.assert_not_called()

    def test_invalid_arguments_keep_argparse_exit_code(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            pipeline.main(["select"])
        self.assertEqual(caught.exception.code, 2)

    def test_actual_workflow_error_is_not_logged_twice(self):
        with self.assertLogs(pipeline.logger, level="ERROR") as logs:
            self.assertEqual(pipeline.main([
                "select", "--days-back", "-1", "--output-filepath", "out"
            ]), 1)
        self.assertEqual(len(logs.records), 1)
        self.assertIsInstance(logs.records[0].exc_info[1], ValueError)

    def test_direct_workflow_preserves_original_failure(self):
        error = OSError("archive unavailable")
        with patch.object(pipeline, "read_metadata_archive", side_effect=error), patch.object(
            pipeline.logger, "exception"
        ) as log, self.assertRaises(OSError) as caught:
            pipeline.get_relevant(1, "unused")
        self.assertIs(caught.exception, error)
        log.assert_not_called()

    def test_empty_embedding_input_returns_none_without_loading_models(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "papers.jsonl"
            source.touch()
            self.assertIsNone(pipeline.embed_papers(source, output_path=root / "out"))


if __name__ == "__main__":
    unittest.main()
