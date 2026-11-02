"""Training label conversion and TSV validation without model dependencies."""

import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from paper_clustering.train.load_data import (
    LABEL_FIELDS, load_labelled_paragraphs, normalize_distribution, print_dataset_statistics,
)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "labels.tsv"

    def write(self, rows, extra_fields=()):
        fields = ["section_header", "paragraph", "label", *extra_fields]
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def test_normalization_and_invalid_distributions(self):
        self.assertEqual(normalize_distribution([0, 0, 1, 1, 0]), [0, 0, 0.5, 0.5, 0])
        for values in ([], [1] * 4, [0] * 5, [-1, 2, 0, 0, 0],
                       [float("nan"), 1, 0, 0, 0], [float("inf"), 1, 0, 0, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_distribution(values)

    def test_human_labels_preserve_paper_ids_and_skip_blank_paragraphs(self):
        self.write([
            {"arxiv_id": " a ", "section_header": " Intro ", "paragraph": " Text ", "label": "4"},
            {"arxiv_id": "b", "section_header": "Other", "paragraph": " ", "label": "bad"},
        ], ["arxiv_id"])
        self.assertEqual(load_labelled_paragraphs(self.path),
                         (["a"], ["Section: Intro Text: Text"], [[0, 0, 0, 1, 0]]))

    def test_soft_labels_take_precedence_and_missing_ids_become_empty_strings(self):
        self.write([{"section_header": "Intro", "paragraph": "Text", "label": "unused",
                     **dict(zip(LABEL_FIELDS, [0, 0, 0, 2, 2]))}], LABEL_FIELDS)
        ids, _, labels = load_labelled_paragraphs(self.path)
        self.assertEqual(ids, [""])
        self.assertEqual(labels, [[0, 0, 0, 0.5, 0.5]])

    def test_invalid_labels_include_line_numbers(self):
        for value in ("0", "6", "1.5", "nan", "bad"):
            self.write([{"section_header": "Intro", "paragraph": "Text", "label": value}])
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "line 2"):
                load_labelled_paragraphs(self.path)

    def test_incomplete_headers_and_empty_datasets_fail(self):
        self.path.write_text("paragraph\tlabel\ntext\t1\n")
        with self.assertRaises(ValueError):
            load_labelled_paragraphs(self.path)
        self.write([], ["probability_A"])
        with self.assertRaisesRegex(ValueError, "all five"):
            load_labelled_paragraphs(self.path)
        self.write([])
        with self.assertRaisesRegex(ValueError, "No labelled"):
            load_labelled_paragraphs(self.path)

    def test_statistics_count_token_limits_and_expected_labels(self):
        tokenizer = Mock(return_value={"input_ids": [[0] * 512, [0] * 514]})
        output = StringIO()
        with redirect_stdout(output):
            print_dataset_statistics(["a", "b"], [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]], tokenizer)
        self.assertIn("513.0 tokens", output.getvalue())
        self.assertEqual(output.getvalue().count("50.0%"), 2)
        self.assertFalse(tokenizer.call_args.kwargs["truncation"])
        with self.assertRaises(ValueError):
            print_dataset_statistics([], [], tokenizer)


if __name__ == "__main__":
    unittest.main()
