#!/usr/bin/env python3
"""Review ambiguous extracted connections: python scripts/review_matches.py RUN_DIR."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

REVIEW_MARKER = "human review needed"
REQUIRED_FIELDS = {
    "connection_id", "arxiv_id", "condition", "reference_source",
    "reference_target", "reference_qualification", "predicted_source",
    "predicted_target", "relation", "qualifications", "evidence",
    "match", "review_notes",
}


def needs_review(row: dict[str, str]) -> bool:
    return (row["match"].strip().lower() in ("", "uncertain")
            or REVIEW_MARKER in row["review_notes"].lower())


def load_review(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read the complete CSV, including columns unrelated to this review."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = REQUIRED_FIELDS - set(fields)
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    identities = [row["connection_id"] for row in rows]
    if len(set(identities)) != len(identities) or any(not key for key in identities):
        raise ValueError("Connection IDs must be unique and nonempty")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row: column count does not match header")
    return fields, rows


def reviewed_notes(notes: str, match: str) -> str:
    """Keep the reason while removing the pending marker from resolved rows."""
    import re

    notes = re.sub(re.escape(REVIEW_MARKER) + r"\s*:?\s*", "", notes,
                   flags=re.IGNORECASE).strip()
    if match == "uncertain":
        return REVIEW_MARKER + (f": {notes}" if notes else "")
    return notes or "Human reviewed."


def save_decision(path: Path, previous: dict[str, str], match: str,
                  notes: str) -> dict[str, str]:
    """Merge one decision into the current file and replace it atomically."""
    if match not in ("yes", "no", "uncertain"):
        raise ValueError(f"Invalid match: {match}")
    fields, rows = load_review(path)
    row = next((item for item in rows
                if item["connection_id"] == previous["connection_id"]), None)
    if row != previous:
        raise ValueError("This candidate changed on disk. Reopen the reviewer to reload it.")
    row.update(match=match, review_notes=reviewed_notes(notes, match))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.chmod(temporary_path, path.stat().st_mode & 0o777)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return dict(row)


def format_fields(row: dict[str, str], fields: tuple[str, ...]) -> str:
    return "\n\n".join(f"{field.replace('_', ' ').title()}\n{row.get(field) or '(not stated)'}"
                        for field in fields)


def evidence_text(row: dict[str, str]) -> str:
    try:
        evidence = json.loads(row["evidence"])
        return "\n\n".join(f"[{item['block_id']}]\n{item['quote']}" for item in evidence)
    except (ValueError, TypeError, KeyError):
        return row["evidence"]


class SourceBlocks:
    """Lazily load saved source blocks without downloads or model calls."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.papers = None

    def for_candidate(self, row: dict[str, str]) -> str:
        if self.papers is None:
            self.papers = {}
            for path in sorted(self.directory.glob("*.json")):
                paper = json.loads(path.read_text(encoding="utf-8"))
                self.papers[paper["arxiv_id"]] = paper
        paper = self.papers.get(row["arxiv_id"])
        if paper is None:
            return f"No saved source found for this paper in {self.directory}."
        evidence_ids = {item["block_id"] for item in json.loads(row["evidence"])}
        blocks = [block for block in paper["blocks"] if block["block_id"] in evidence_ids]
        text = "\n\n".join(f"[{block['block_id']}] {block.get('heading', '')}\n{block['text']}"
                            for block in blocks)
        return f"{paper.get('title', '')}\n\n{text or 'No cited blocks found.'}"


class MatchReviewApp:
    """Review a fixed session queue, saving each choice before advancing."""

    def __init__(self, root, path: Path, rows: list[dict[str, str]]):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.path, self.rows = root, path, rows
        self.index = 0
        self.visited = set()
        self.sources = SourceBlocks(path.parent / "sources")
        self.configure_fonts(root)
        root.title("Connection Match Reviewer")
        root.geometry("1150x850")
        root.minsize(850, 650)
        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        self.status = tk.StringVar()
        ttk.Label(outer, textvariable=self.status).pack(anchor="w")
        ttk.Label(outer, wraplength=1050, text=(
            "Match the annotated source, target, relation, and restrictions. "
            "Field class is scored separately. Yes/No resolves the case; "
            "Uncertain keeps it pending. Each decision saves immediately."
        )).pack(fill="x", pady=8)
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        comparison = ttk.Frame(self.tabs)
        self.tabs.add(comparison, text="Reference and candidate")
        comparison.columnconfigure((0, 1), weight=1)
        comparison.rowconfigure(0, weight=1)
        self.reference = self.text_panel(comparison)
        self.reference.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.candidate = self.text_panel(comparison)
        self.candidate.grid(row=0, column=1, sticky="nsew")
        self.evidence = self.text_panel(self.tabs)
        self.tabs.add(self.evidence, text="Quoted evidence")
        self.context = self.text_panel(self.tabs)
        self.tabs.add(self.context, text="Full cited source blocks")
        self.tabs.bind("<<NotebookTabChanged>>", self.show_context)
        ttk.Label(outer, text="Review notes (saved with your decision)").pack(anchor="w", pady=(8, 2))
        self.notes = self.text_panel(outer, height=4)
        self.notes.pack(fill="x")
        self.add_controls(outer)
        ttk.Label(outer, text=str(path), foreground="#555555").pack(anchor="w", pady=(8, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.show_candidate()

    def configure_fonts(self, root):
        from tkinter import font as tkfont

        available = set(tkfont.families(root))
        reading_family = next((name for name in (
            "Times New Roman", "Liberation Serif", "DejaVu Serif"
        ) if name in available), "Times")
        control_family = next((name for name in (
            "Arial", "Liberation Sans", "DejaVu Sans"
        ) if name in available), "Helvetica")
        self.reading_font = tkfont.Font(root=root, family=reading_family, size=14)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name, root=root).configure(family=control_family, size=11)

    def text_panel(self, parent, height=12):
        from tkinter.scrolledtext import ScrolledText

        return ScrolledText(parent, wrap="word", width=45, height=height,
                            font=self.reading_font, spacing1=2, spacing3=4,
                            padx=10, pady=10, undo=True)

    def add_controls(self, parent):
        from tkinter import ttk

        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(10, 0))
        for label, value, key in (("Yes", "yes", "y"), ("No", "no", "n"),
                                  ("Uncertain", "uncertain", "u")):
            ttk.Button(controls, text=f"{label} (Ctrl+{key.upper()})",
                       command=lambda choice=value: self.decide(choice)).pack(side="left", padx=3)
            self.root.bind(f"<Control-{key}>", lambda event, choice=value: self.decide(choice))
        for label, offset in (("Previous", -1), ("Skip / Next", 1)):
            ttk.Button(controls, text=label,
                       command=lambda step=offset: self.navigate(step)).pack(side="right", padx=3)

    @staticmethod
    def set_text(widget, text, readonly=True):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.edit_reset()
        if readonly:
            widget.configure(state="disabled")

    def show_candidate(self):
        row = self.rows[self.index]
        pending = sum(needs_review(item) for item in self.rows)
        self.status.set(f"{self.index + 1}/{len(self.rows)} | {row['connection_id']} | "
                        f"Current: {row['match'] or 'unreviewed'} | {pending} pending in session")
        self.set_text(self.reference, format_fields(row, (
            "reference_source", "reference_target", "reference_qualification", "reference_category")))
        self.set_text(self.candidate, format_fields(row, (
            "predicted_source", "predicted_target", "relation", "qualifications", "predicted_category")))
        self.set_text(self.evidence, evidence_text(row))
        self.set_text(self.context, "Select this tab to load the saved source blocks.")
        self.set_text(self.notes, row["review_notes"], readonly=False)
        self.show_context()

    def show_context(self, event=None):
        if self.tabs.index(self.tabs.select()) != 2:
            return
        try:
            text = self.sources.for_candidate(self.rows[self.index])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            text = f"Could not load saved source: {exc}"
        self.set_text(self.context, text)

    def discard_edits(self) -> bool:
        from tkinter import messagebox

        if self.notes.get("1.0", "end-1c") == self.rows[self.index]["review_notes"]:
            return True
        return messagebox.askyesno("Unsaved notes", "Discard notes edited since the last decision?",
                                   parent=self.root)

    def navigate(self, offset):
        index = max(0, min(len(self.rows) - 1, self.index + offset))
        if index != self.index and self.discard_edits():
            self.index = index
            self.show_candidate()

    def decide(self, match):
        from tkinter import messagebox

        try:
            self.rows[self.index] = save_decision(
                self.path, self.rows[self.index], match, self.notes.get("1.0", "end-1c"))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Decision not saved", str(exc), parent=self.root)
            return "break"
        self.visited.add(self.index)
        order = list(range(self.index + 1, len(self.rows))) + list(range(self.index))
        next_index = next((i for i in order if i not in self.visited), None)
        if next_index is not None:
            self.index = next_index
        self.show_candidate()
        if next_index is None:
            messagebox.showinfo("Session reviewed", "Every case in this session has a saved decision. "
                                "Uncertain cases remain available when you reopen the reviewer. "
                                "Use Previous/Next to revise decisions.", parent=self.root)
        return "break"

    def close(self):
        if self.discard_edits():
            self.root.destroy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="match_review.csv or its run directory")
    parser.add_argument("--all", action="store_true", help="Include resolved cases for revision")
    args = parser.parse_args(argv)
    path = args.input / "match_review.csv" if args.input.is_dir() else args.input
    try:
        _, rows = load_review(path)
        rows = [row for row in rows if args.all or needs_review(row)]
        if not rows:
            print("No cases need review. Use --all to revise resolved decisions.")
            return 0
        import tkinter as tk
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f"error: could not open Tk GUI: {exc}\nRun from a graphical desktop with Tkinter installed.\n")
    MatchReviewApp(root, path.resolve(), rows)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
