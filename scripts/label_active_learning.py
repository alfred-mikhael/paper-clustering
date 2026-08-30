#!/usr/bin/env python3
"""Interactively label paragraphs selected for active learning."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_clustering.label.paragraph_label_gui import ParagraphLabelApp

IDENTITY_FIELDS = ("arxiv_id", "section_index", "paragraph_index")
REQUIRED_FIELDS = (*IDENTITY_FIELDS, "section_header", "paragraph", "strategy")
LABEL_FIELD = "label"


@dataclass(frozen=True)
class SelectedParagraph:
    """A paragraph selected by an active-learning strategy."""

    row: dict[str, str]

    @property
    def key(self) -> tuple[str, int, int]:
        return (
            self.row["arxiv_id"],
            int(self.row["section_index"]),
            int(self.row["paragraph_index"]),
        )

    @property
    def arxiv_id(self) -> str:
        return self.row["arxiv_id"]

    @property
    def section_header(self) -> str:
        return self.row["section_header"]

    @property
    def paragraph(self) -> str:
        return self.row["paragraph"]


def load_selected(path: Path) -> tuple[list[str], list[SelectedParagraph]]:
    """Load uniquely identified selected paragraphs from a TSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        missing = sorted(set(REQUIRED_FIELDS) - set(fieldnames))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        samples = []
        seen_keys = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                sample = SelectedParagraph(dict(row))
                key = sample.key
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid paragraph identity on line {line_number}"
                ) from exc
            if not key[0] or key in seen_keys:
                raise ValueError(
                    f"Duplicate or empty paragraph identity on line {line_number}"
                )
            seen_keys.add(key)
            samples.append(sample)
    if not samples:
        raise ValueError(f"No selected paragraphs found in {path}")
    return fieldnames, samples


def load_labels(path: Path) -> dict[tuple[str, int, int], int]:
    """Read validated, previously saved labels, if present."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not set((*IDENTITY_FIELDS, LABEL_FIELD)).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} is not an active-learning label TSV")
        labels = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                label = int(row[LABEL_FIELD])
                key = (
                    row["arxiv_id"],
                    int(row["section_index"]),
                    int(row["paragraph_index"]),
                )
                if label not in range(1, 6):
                    raise ValueError
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label on line {line_number}") from exc
            labels[key] = label
    return labels


def save_labels(
    path: Path,
    fieldnames: list[str],
    samples: list[SelectedParagraph],
    labels: dict[tuple[str, int, int], int],
) -> None:
    """Atomically save labelled samples while preserving selection diagnostics."""
    output_fields = [field for field in fieldnames if field != LABEL_FIELD]
    output_fields.append(LABEL_FIELD)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, delimiter="\t")
        writer.writeheader()
        for sample in samples:
            label = labels.get(sample.key)
            if label is None:
                continue
            writer.writerow({**sample.row, LABEL_FIELD: label})
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="TSV written by select_active_learning_samples.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="labelled TSV path (default: INPUT with _labels appended)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output or args.input.with_name(f"{args.input.stem}_labels.tsv")
    if output_path.resolve() == args.input.resolve():
        raise SystemExit("error: --output must differ from the selected input TSV")
    try:
        fieldnames, samples = load_selected(args.input)
        import tkinter as tk
    except (ImportError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    root = tk.Tk()

    def details(sample: SelectedParagraph) -> str:
        diagnostics = [f"Strategy: {sample.row['strategy']}"]
        for field in ("predicted_score", "entropy", "weighted_emd"):
            if sample.row.get(field):
                diagnostics.append(f"{field.replace('_', ' ')}: {sample.row[field]}")
        return "  •  ".join(diagnostics)

    ParagraphLabelApp(
        root,
        samples,
        load_labels(output_path),
        lambda labels: save_labels(output_path, fieldnames, samples, labels),
        title="Active-Learning Paragraph Labeller",
        status=f"Labels are saved to {output_path}",
        details=details,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
