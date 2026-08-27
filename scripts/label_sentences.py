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
import queue
import re
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

# Make ``src`` importable when this file is run as ``python scripts/...``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_clustering.extract_text import ArxivSection, get_paper

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
        )
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


class ParagraphLabelApp:
    def __init__(
        self,
        root,
        arxiv_ids: list[str],
        output_path: Path,
        cache_dir: Path,
        refresh: bool,
    ) -> None:
        # Tk is imported lazily so extraction helpers can be tested headlessly.
        from tkinter import BOTH, LEFT, RIGHT, X, Button, Frame, Label, StringVar, Text
        from tkinter import font as tkfont
        from tkinter import ttk

        self.root = root
        self.arxiv_ids = arxiv_ids
        self.output_path = output_path
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.samples: list[ParagraphSample] = []
        self.labels = load_labels(output_path)
        self.index = 0
        self.loading_queue: queue.Queue = queue.Queue()

        root.title("arXiv Paragraph Labeller")
        root.geometry("940x650")
        root.minsize(700, 500)

        available_fonts = set(tkfont.families(root))
        interface_family = (
            "Arial"
            if "Arial" in available_fonts
            else (
                "Liberation Sans"
                if "Liberation Sans" in available_fonts
                else "Helvetica"
            )
        )
        tkfont.nametofont("TkDefaultFont").configure(family=interface_family, size=11)
        tkfont.nametofont("TkTextFont").configure(family=interface_family, size=14)
        self.paragraph_font = tkfont.Font(root=root, family=interface_family, size=14)

        outer = ttk.Frame(root, padding=18)
        outer.pack(fill=BOTH, expand=True)
        self.location_var = StringVar(value="Preparing papers...")
        self.progress_var = StringVar(value="")
        ttk.Label(
            outer, textvariable=self.location_var, font=("TkDefaultFont", 11, "bold")
        ).pack(anchor="w")

        progress_row = ttk.Frame(outer)
        progress_row.pack(fill=X, pady=(8, 18))
        self.progress = ttk.Progressbar(progress_row, mode="determinate")
        self.progress.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(
            progress_row, textvariable=self.progress_var, width=28, anchor="e"
        ).pack(side=RIGHT, padx=(10, 0))

        self.section_var = StringVar(value="")
        ttk.Label(outer, text="Section", foreground="#555555").pack(anchor="w")
        ttk.Label(
            outer,
            textvariable=self.section_var,
            font=("TkDefaultFont", 13, "bold"),
            wraplength=880,
            justify="left",
        ).pack(fill=X, pady=(2, 14))

        self.paragraph_text = Text(
            outer,
            # Keep enough requested space for the controls below. ``expand``
            # still lets the reading area consume spare space in larger windows.
            height=10,
            wrap="word",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10,
            font=self.paragraph_font,
            background=outer.winfo_toplevel().cget("background"),
            cursor="arrow",
        )
        self.paragraph_text.pack(fill=BOTH, expand=True, pady=(2, 11))
        self.paragraph_text.configure(state="disabled")

        controls = Frame(outer)
        controls.pack(fill=X, pady=(18, 0))
        Label(controls, text="Label:").pack(side=LEFT, padx=(0, 8))
        self.label_buttons = []
        for value in range(1, 6):
            button = Button(
                controls,
                text=str(value),
                width=4,
                command=lambda selected=value: self.set_label(selected),
            )
            button.pack(side=LEFT, padx=3)
            self.label_buttons.append(button)
        ttk.Button(
            controls, text="Next unlabelled (Enter)", command=self.next_unlabelled
        ).pack(side=RIGHT)
        ttk.Button(controls, text="Next  →", command=self.go_next).pack(
            side=RIGHT, padx=5
        )
        ttk.Button(controls, text="←  Previous", command=self.go_previous).pack(
            side=RIGHT, padx=5
        )

        self.status_var = StringVar(value="Downloading and extracting papers...")
        ttk.Label(outer, textvariable=self.status_var, foreground="#555555").pack(
            anchor="w", pady=(14, 0)
        )

        root.bind("<Left>", self.go_previous)
        root.bind("<Right>", self.go_next)
        root.bind("<Return>", self.next_unlabelled)
        root.bind("<KP_Enter>", self.next_unlabelled)
        for value in range(1, 6):
            root.bind(
                str(value), lambda _event, selected=value: self.set_label(selected)
            )

        self._set_controls_enabled(False)
        threading.Thread(target=self._load_papers, daemon=True).start()
        root.after(100, self._poll_loading_queue)

    def _load_papers(self) -> None:
        all_samples: list[ParagraphSample] = []
        errors: list[str] = []
        for number, arxiv_id in enumerate(self.arxiv_ids, 1):
            self.loading_queue.put(
                ("status", f"Loading {arxiv_id} ({number}/{len(self.arxiv_ids)})...")
            )
            try:
                all_samples.extend(
                    fetch_samples(arxiv_id, self.cache_dir, self.refresh)
                )
            except Exception as exc:  # keep usable papers even if one fails
                errors.append(f"{arxiv_id}: {exc}")
        self.loading_queue.put(("done", all_samples, errors))

    def _poll_loading_queue(self) -> None:
        try:
            while True:
                message = self.loading_queue.get_nowait()
                if message[0] == "status":
                    self.status_var.set(message[1])
                else:
                    self._finish_loading(message[1], message[2])
                    return
        except queue.Empty:
            self.root.after(100, self._poll_loading_queue)

    def _finish_loading(
        self, samples: list[ParagraphSample], errors: list[str]
    ) -> None:
        from tkinter import messagebox

        self.samples = samples
        if errors:
            messagebox.showwarning("Some papers were skipped", "\n".join(errors))
        if not samples:
            messagebox.showerror("Nothing to label", "No paragraphs could be loaded.")
            self.root.destroy()
            return
        self._set_controls_enabled(True)
        first_unlabelled = self._find_unlabelled(start=-1, wrap=True)
        self.index = first_unlabelled if first_unlabelled is not None else 0
        self.status_var.set(
            f"Labels are saved after every choice to {self.output_path}"
        )
        self.show_sample()

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.label_buttons:
            button.configure(state=state)

    def _set_paragraph(self, sample: ParagraphSample) -> None:
        """Render one paragraph in the reading panel."""
        self.paragraph_text.configure(state="normal")
        self.paragraph_text.delete("1.0", "end")
        self.paragraph_text.insert("end", sample.paragraph)
        self.paragraph_text.configure(state="disabled")

    def show_sample(self) -> None:
        sample = self.samples[self.index]
        current_label = self.labels.get(sample.key)
        self.location_var.set(
            f"{sample.arxiv_id}  •  paragraph {self.index + 1} of {len(self.samples)}"
        )
        self.section_var.set(sample.section_header)
        self._set_paragraph(sample)
        for value, button in enumerate(self.label_buttons, 1):
            button.configure(relief="sunken" if value == current_label else "raised")
        labelled = sum(sample.key in self.labels for sample in self.samples)
        remaining = len(self.samples) - labelled
        self.progress.configure(maximum=len(self.samples), value=labelled)
        self.progress_var.set(f"{labelled} labelled • {remaining} remaining")

    def set_label(self, value: int):
        if not self.samples:
            return "break"
        self.labels[self.samples[self.index].key] = value
        save_labels(self.output_path, self.samples, self.labels)
        next_index = self._find_unlabelled(start=self.index, wrap=True)
        if next_index is None:
            self.status_var.set(
                "All paragraphs are labelled. You can still review or change labels."
            )
            self.show_sample()
        else:
            self.index = next_index
            self.show_sample()
        return "break"

    def _find_unlabelled(self, start: int, wrap: bool) -> int | None:
        if not self.samples:
            return None
        indices = list(range(start + 1, len(self.samples)))
        if wrap and start > 0:
            # Do not select the current sample again when wrapping.
            indices += list(range(0, start))
        return next(
            (i for i in indices if self.samples[i].key not in self.labels), None
        )

    def next_unlabelled(self, _event=None):
        if not self.samples:
            return "break"
        next_index = self._find_unlabelled(start=self.index, wrap=True)
        if next_index is None:
            remaining = sum(sample.key not in self.labels for sample in self.samples)
            self.status_var.set(
                "All paragraphs are labelled."
                if remaining == 0
                else "There are no other unlabelled paragraphs."
            )
        else:
            self.index = next_index
            self.show_sample()
        return "break"

    def go_previous(self, _event=None):
        if self.samples:
            self.index = max(0, self.index - 1)
            self.show_sample()
        return "break"

    def go_next(self, _event=None):
        if self.samples:
            self.index = min(len(self.samples) - 1, self.index + 1)
            self.show_sample()
        return "break"


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

    try:
        import tkinter as tk
    except ImportError:
        parser.error("Tkinter is unavailable; install your system's python3-tk package")
    root = tk.Tk()
    ParagraphLabelApp(root, arxiv_ids, args.output, args.cache_dir, args.refresh)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
