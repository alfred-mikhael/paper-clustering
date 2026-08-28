"""Load and summarize labelled paragraph datasets."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path


LABEL_FIELDS = tuple(f"probability_{label}" for label in "ABCDE")


def normalize_distribution(values: Sequence[float]) -> list[float]:
    distribution = [float(value) for value in values]
    if (
        len(distribution) != 5
        or any(not math.isfinite(value) or value < 0 for value in distribution)
        or sum(distribution) <= 0
    ):
        raise ValueError("labels must be length-5 probability distributions")
    total = sum(distribution)
    return [value / total for value in distribution]


def load_labelled_paragraphs(
    path: str | Path,
) -> tuple[list[str], list[str], list[list[float]]]:
    """Load paper IDs, model inputs, and A-E label distributions from a TSV.

    Human ``label`` values are converted to one-hot distributions. If all five
    ``probability_A`` through ``probability_E`` fields are present, those soft
    labels are normalized and returned instead. The paper-ID list is empty
    strings when the TSV does not contain an ``arxiv_id`` column, which keeps
    the loader usable for training-only datasets.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        if not {"section_header", "paragraph", "label"}.issubset(fields):
            raise ValueError(f"{path} must contain paragraph and label columns")
        probability_fields = fields.intersection(LABEL_FIELDS)
        if probability_fields and probability_fields != set(LABEL_FIELDS):
            raise ValueError(f"{path} must contain all five probability_A-E columns")
        has_probabilities = bool(probability_fields)

        paper_ids = []
        paragraphs = []
        labels = []
        for line_number, row in enumerate(reader, start=2):
            section = row["section_header"].strip()
            paragraph = row["paragraph"].strip()
            if not paragraph:
                continue
            try:
                if has_probabilities:
                    distribution = normalize_distribution(
                        [row[field] for field in LABEL_FIELDS]
                    )
                else:
                    human_label = float(row["label"])
                    if not human_label.is_integer() or not 1 <= human_label <= 5:
                        raise ValueError
                    distribution = [0.0] * 5
                    distribution[int(human_label) - 1] = 1.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label on line {line_number}") from exc
            paper_ids.append(row.get("arxiv_id", "").strip())
            paragraphs.append(f"Section: {section} Text: {paragraph}")
            labels.append(distribution)

    if not paragraphs:
        raise ValueError(f"No labelled paragraphs found in {path}")
    return paper_ids, paragraphs, labels


def print_dataset_statistics(
    paragraphs: Sequence[str],
    labels: Sequence[Sequence[float]],
    tokenizer,
) -> None:
    """Print token-length and label-distribution summary statistics."""
    if len(paragraphs) != len(labels) or not paragraphs:
        raise ValueError("paragraphs and labels must be nonempty and equally sized")

    tokenized = tokenizer(
        list(paragraphs),
        add_special_tokens=True,
        padding=False,
        truncation=False,
        verbose=False,
    )
    lengths = [len(input_ids) for input_ids in tokenized["input_ids"]]
    expected_scores = [
        sum(score * probability for score, probability in enumerate(label, start=1))
        for label in labels
    ]

    average_length = sum(lengths) / len(lengths)
    over_limit = 100 * sum(length > 512 for length in lengths) / len(lengths)
    at_least_c = 100 * sum(score >= 3 for score in expected_scores) / len(labels)
    print("Dataset statistics:")
    print(f"  {len(paragraphs)} training paragraphs")
    print(f"  Average paragraph length: {average_length:.1f} tokens")
    print(f"  Paragraphs exceeding 512 tokens: {over_limit:.1f}%")
    print(f"  Labels with expected score >= C: {at_least_c:.1f}%")
