#!/usr/bin/env python3
"""Interactive paragraph labeller for arXiv papers.

Downloads arXiv TeX sources, converts their sections to plain text, and shows
one paragraph at a time in a small Tk GUI. Labels are written to a TSV file
after every change, making an interrupted session resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

# Make ``src`` importable when this file is run as ``python scripts/...``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_clustering.extract_text import ArxivSection, get_paper
from paper_clustering.label.paragraph_label_gui import ParagraphLabelApp

ARXIV_ID_RE = re.compile(
    r"^(?:https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/)?"
    r"(?P<id>(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)"
    r"(?:\.pdf)?$",
    re.IGNORECASE,
)
LABEL_FIELDS = (
    "arxiv_id",
    "section_index",
    "paragraph_index",
    "section_header",
    "paragraph",
    "label",
)
SAMPLE_CACHE_VERSION = 1


@dataclass(frozen=True)
class ParagraphSample:
    arxiv_id: str
    section_index: int
    paragraph_index: int
    section_header: str
    paragraph: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.arxiv_id, self.section_index, self.paragraph_index


def normalize_arxiv_id(value: str) -> str:
    """Accept a bare arXiv ID or an arxiv.org abs/PDF URL."""
    candidate = value.strip().rstrip("/")
    match = ARXIV_ID_RE.fullmatch(candidate)
    if not match:
        raise ValueError(f"Invalid arXiv ID or URL: {value!r}")
    return match.group("id")


def samples_from_sections(sections: list[ArxivSection]) -> list[ParagraphSample]:
    """Flatten the paragraphs returned by ``get_sections`` in document order."""
    samples: list[ParagraphSample] = []
    for section in sections:
        for paragraph_index, paragraph in enumerate(section.text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            samples.append(
                ParagraphSample(
                    arxiv_id=section.arxiv_id,
                    section_index=section.section_index,
                    paragraph_index=paragraph_index,
                    section_header=section.section_header or "(Untitled section)",
                    paragraph=paragraph,
                )
            )
    return samples


def fetch_samples(
    arxiv_id: str, cache_dir: Path, refresh: bool = False
) -> list[ParagraphSample]:
    """Load processed samples from cache or download and process arXiv source."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{arxiv_id.replace('/', '_')}.json"
    if cache_path.exists() and not refresh:
        try:
            with cache_path.open(encoding="utf-8") as handle:
                cached = json.load(handle)
            if (
                isinstance(cached, dict)
                and cached.get("cache_version") == SAMPLE_CACHE_VERSION
            ):
                return [ParagraphSample(**item) for item in cached["samples"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Treat a partial or incompatible cache as a miss. The successful
            # download below atomically replaces it with a valid cache.
            pass

    with requests.Session() as session:
        sections = get_paper(
            session,
            arxiv_id,
            # r"(Overview|Outline|Introduction|Review|Survey|Conclu.*|Discussion|Tech.*)",
        ).sections
    samples = samples_from_sections(sections)
    if not samples:
        raise RuntimeError(
            "no section paragraphs could be extracted from the TeX source"
        )

    temporary_path = cache_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_version": SAMPLE_CACHE_VERSION,
                "samples": [asdict(sample) for sample in samples],
            },
            handle,
            ensure_ascii=False,
        )
    temporary_path.replace(cache_path)
    return samples


def _load_label_rows(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    rows: dict[tuple[str, int, int], dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                label = int(row["label"])
                if label not in range(1, 6):
                    raise ValueError
                key = (
                    row["arxiv_id"],
                    int(row["section_index"]),
                    int(row["paragraph_index"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed label row in {path}: {row}") from exc
            rows[key] = {field: row[field] for field in LABEL_FIELDS}
    return rows


def load_labels(path: Path) -> dict[tuple[str, int, int], int]:
    return {key: int(row["label"]) for key, row in _load_label_rows(path).items()}


def save_labels(
    path: Path,
    samples: list[ParagraphSample],
    labels: dict[tuple[str, int, int], int],
) -> None:
    """Atomically write labels in document order, one row per labelled sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_label_rows(path)
    current_keys = {sample.key for sample in samples}
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS, delimiter="\t")
        writer.writeheader()
        # A user may reopen the same output with only a subset of the original
        # paper IDs. Preserve records outside the active session.
        for key, row in existing_rows.items():
            if key not in current_keys:
                writer.writerow(row)
        for sample in samples:
            if sample.key not in labels:
                continue
            writer.writerow(
                {
                    "arxiv_id": sample.arxiv_id,
                    "section_index": sample.section_index,
                    "paragraph_index": sample.paragraph_index,
                    "section_header": sample.section_header,
                    "paragraph": sample.paragraph,
                    "label": labels[sample.key],
                }
            )
    temporary_path.replace(path)


def parse_id_file(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.split("#", 1)[0]
        values.extend(part for part in re.split(r"[\s,]+", content) if part)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download arXiv papers and label their paragraphs from 1 to 5."
    )
    parser.add_argument("arxiv_ids", nargs="*", help="arXiv IDs or abs/PDF URLs")
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="text file containing whitespace-, comma-, or newline-separated IDs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paragraph_labels.tsv"),
        help="label TSV path (default: paragraph_labels.tsv)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".paragraph_label_cache"),
        help="processed-paper cache (default: .paragraph_label_cache)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="redownload papers instead of using cache",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_ids = list(args.arxiv_ids)
    if args.ids_file:
        try:
            raw_ids.extend(parse_id_file(args.ids_file))
        except OSError as exc:
            parser.error(str(exc))
    if not raw_ids:
        parser.error("provide at least one arXiv ID or use --ids-file")
    try:
        arxiv_ids = list(dict.fromkeys(normalize_arxiv_id(value) for value in raw_ids))
    except ValueError as exc:
        parser.error(str(exc))

    samples: list[ParagraphSample] = []
    errors: list[str] = []
    for number, arxiv_id in enumerate(arxiv_ids, start=1):
        print(f"Loading {arxiv_id} ({number}/{len(arxiv_ids)})...", file=sys.stderr)
        try:
            samples.extend(fetch_samples(arxiv_id, args.cache_dir, args.refresh))
        except Exception as exc:  # Keep usable papers even if one fails.
            errors.append(f"{arxiv_id}: {exc}")
    if not samples:
        parser.error("No paragraphs could be loaded.\n" + "\n".join(errors))

    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        parser.error("Tkinter is unavailable; install your system's python3-tk package")
    root = tk.Tk()
    if errors:
        messagebox.showwarning("Some papers were skipped", "\n".join(errors))
    ParagraphLabelApp(
        root,
        samples,
        load_labels(args.output),
        lambda labels: save_labels(args.output, samples, labels),
        title="arXiv Paragraph Labeller",
        status=f"Labels are saved after every choice to {args.output}",
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
