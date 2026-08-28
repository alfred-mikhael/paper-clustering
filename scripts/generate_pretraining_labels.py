#!/usr/bin/env python3
"""Generate a resumable weakly labelled paragraph dataset from arXiv papers."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
import time
from pathlib import Path

import requests

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_clustering.extract_text import get_paper
from paper_clustering.label.slm_labels import LABELS, SLMWeakLabelGen

OUTPUT_FIELDS = (
    "arxiv_id",
    "section_index",
    "paragraph_index",
    "section_header",
    "paragraph",
    "label",
    *(f"probability_{label}" for label in LABELS),
)

ARXIV_API_WAIT_TIME = 3


def base_arxiv_id(arxiv_id: str) -> str:
    """Remove a trailing arXiv version so unversioned inputs resume correctly."""
    return re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)


def parse_ids(path: Path) -> list[str]:
    """Read unique arXiv IDs separated by commas or whitespace."""
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.split("#", 1)[0]
        values.extend(part for part in re.split(r"[\s,]+", content) if part)
    return list(dict.fromkeys(values))


def load_existing_rows(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    """Load an existing output so completed papers can be skipped."""
    if not path.exists():
        return {}

    rows: dict[tuple[str, int, int], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not set(OUTPUT_FIELDS).issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{path} is not a pretraining-label TSV with the expected columns"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                key = (
                    row["arxiv_id"],
                    int(row["section_index"]),
                    int(row["paragraph_index"]),
                )
                rows[key] = {field: row[field] for field in OUTPUT_FIELDS}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed row {line_number} in {path}") from exc
    return rows


def save_rows(path: Path, rows: dict[tuple[str, int, int], dict[str, str]]) -> None:
    """Atomically save every generated paragraph label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows.values())
    temporary_path.replace(path)


def rows_for_paper(paper, labels) -> list[dict[str, str]]:
    """Pair document-order SLM probabilities with their source paragraphs."""
    paragraphs = [
        (section, paragraph_index, paragraph)
        for section in paper.sections
        for paragraph_index, paragraph in enumerate(section.text)
    ]
    if len(paragraphs) != len(labels):
        raise ValueError(
            f"Expected {len(paragraphs)} labels for {paper.metadata.arxiv_id}, "
            f"received {len(labels)}"
        )

    rows = []
    for (section, paragraph_index, paragraph), probabilities in zip(paragraphs, labels):
        if hasattr(probabilities, "detach"):
            probabilities = probabilities.detach().cpu().tolist()
        else:
            probabilities = list(probabilities)
        probabilities = [float(value) for value in probabilities]
        if (
            len(probabilities) != len(LABELS)
            or not all(math.isfinite(value) and value >= 0 for value in probabilities)
            or sum(probabilities) <= 0
        ):
            raise ValueError(
                f"Invalid A-E probability vector for {paper.metadata.arxiv_id}"
            )

        total = sum(probabilities)
        probabilities = [value / total for value in probabilities]
        # Keep the expected rating as a human-readable summary. Training uses
        # the complete probability_A through probability_E distribution below.
        expected_label = sum(
            rating * probability
            for rating, probability in enumerate(probabilities, start=1)
        )
        rows.append(
            {
                "arxiv_id": section.arxiv_id,
                "section_index": str(section.section_index),
                "paragraph_index": str(paragraph_index),
                "section_header": section.section_header,
                "paragraph": paragraph,
                "label": f"{expected_label:.9g}",
                **{
                    f"probability_{label}": f"{probability:.9g}"
                    for label, probability in zip(LABELS, probabilities)
                },
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate weak paragraph labels for SciBERT pretraining"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="file containing comma- or whitespace-separated arXiv IDs",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("pretraining_labels.tsv"),
        help="output TSV (default: pretraining_labels.tsv)",
    )
    parser.add_argument("--server-url", help="llama.cpp server URL")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--cache-dir", type=Path, help="SLM label cache directory")
    parser.add_argument(
        "--retry-completed",
        action="store_true",
        help="process paper IDs that already occur in the output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        arxiv_ids = parse_ids(args.input_path)
        rows = load_existing_rows(args.output_path)
    except (OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 2
    if not arxiv_ids:
        logging.error("No arXiv IDs found in %s", args.input_path)
        return 2

    completed_ids = {base_arxiv_id(key[0]) for key in rows}
    slm = SLMWeakLabelGen(
        server_url=args.server_url,
        request_timeout=args.request_timeout,
        cache_dir=args.cache_dir,
    )
    failures = 0
    try:
        with requests.Session() as session:
            for index, arxiv_id in enumerate(arxiv_ids, start=1):
                if (
                    not args.retry_completed
                    and base_arxiv_id(arxiv_id) in completed_ids
                ):
                    logging.info(
                        "[%d/%d] Skipping completed %s",
                        index,
                        len(arxiv_ids),
                        arxiv_id,
                    )
                    continue
                time.sleep(ARXIV_API_WAIT_TIME)
                logging.info("[%d/%d] Labeling %s", index, len(arxiv_ids), arxiv_id)
                try:
                    paper = get_paper(session, arxiv_id)
                    if not paper.sections:
                        raise RuntimeError("no paragraphs were extracted")
                    paper_rows = rows_for_paper(paper, slm.label_paper(paper))
                    for row in paper_rows:
                        key = (
                            row["arxiv_id"],
                            int(row["section_index"]),
                            int(row["paragraph_index"]),
                        )
                        rows[key] = row
                    save_rows(args.output_path, rows)
                    completed_ids.add(base_arxiv_id(paper.metadata.arxiv_id))
                    logging.info(
                        "Saved %d labels for %s",
                        len(paper_rows),
                        paper.metadata.arxiv_id,
                    )
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    requests.RequestException,
                ) as exc:
                    failures += 1
                    logging.error("Failed to label %s: %s", arxiv_id, exc)
    except KeyboardInterrupt:
        logging.warning(
            "Interrupted; completed papers are saved in %s", args.output_path
        )
        return 130
    finally:
        slm.session.close()

    logging.info(
        "Finished with %d labelled paragraphs and %d failed papers",
        len(rows),
        failures,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
