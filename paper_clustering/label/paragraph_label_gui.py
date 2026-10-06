"""Reusable Tk interface for assigning 1–5 labels to paragraphs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol


ParagraphKey = tuple[str, int, int]


class LabelableParagraph(Protocol):
    """Minimal paragraph interface required by :class:`ParagraphLabelApp`."""

    arxiv_id: str
    section_header: str
    paragraph: str
    key: ParagraphKey


class ParagraphLabelApp:
    """A resumable Tk paragraph labeler with keyboard navigation."""

    def __init__(
        self,
        root,
        samples: Sequence[LabelableParagraph],
        labels: dict[ParagraphKey, int],
        save: Callable[[dict[ParagraphKey, int]], None],
        *,
        title: str,
        status: str,
        details: Callable[[LabelableParagraph], str] | None = None,
    ) -> None:
        from tkinter import BOTH, LEFT, RIGHT, X, Button, Frame, Label, StringVar, Text
        from tkinter import font as tkfont
        from tkinter import ttk

        if not samples:
            raise ValueError("samples cannot be empty")

        self.root = root
        self.samples = list(samples)
        self.labels = labels
        self.save = save
        self.details = details or (lambda _sample: "")
        self.index = self._find_unlabelled(start=-1, wrap=True) or 0

        root.title(title)
        root.geometry("940x700")
        root.minsize(700, 500)

        available_fonts = set(tkfont.families(root))
        family = "Arial" if "Arial" in available_fonts else "Helvetica"
        tkfont.nametofont("TkDefaultFont").configure(family=family, size=11)
        self.paragraph_font = tkfont.Font(root=root, family=family, size=14)

        outer = ttk.Frame(root, padding=18)
        outer.pack(fill=BOTH, expand=True)
        self.location_var = StringVar()
        self.details_var = StringVar()
        self.progress_var = StringVar()
        self.status_var = StringVar(value=status)
        ttk.Label(outer, textvariable=self.location_var, font=(family, 11, "bold")).pack(
            anchor="w"
        )
        ttk.Label(outer, textvariable=self.details_var, foreground="#555555").pack(
            anchor="w", pady=(3, 8)
        )

        progress_row = ttk.Frame(outer)
        progress_row.pack(fill=X, pady=(0, 16))
        self.progress = ttk.Progressbar(progress_row, mode="determinate")
        self.progress.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(progress_row, textvariable=self.progress_var, width=26, anchor="e").pack(
            side=RIGHT, padx=(10, 0)
        )

        self.section_var = StringVar()
        ttk.Label(outer, text="Section", foreground="#555555").pack(anchor="w")
        ttk.Label(
            outer,
            textvariable=self.section_var,
            font=(family, 13, "bold"),
            wraplength=880,
            justify="left",
        ).pack(fill=X, pady=(2, 12))

        self.paragraph_text = Text(
            outer,
            height=11,
            wrap="word",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10,
            font=self.paragraph_font,
            background=root.cget("background"),
            cursor="arrow",
        )
        self.paragraph_text.pack(fill=BOTH, expand=True, pady=(2, 10))
        self.paragraph_text.configure(state="disabled")

        controls = Frame(outer)
        controls.pack(fill=X, pady=(14, 0))
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
        ttk.Button(controls, text="Next unlabelled (Enter)", command=self.next_unlabelled).pack(
            side=RIGHT
        )
        ttk.Button(controls, text="Next  →", command=self.go_next).pack(side=RIGHT, padx=5)
        ttk.Button(controls, text="←  Previous", command=self.go_previous).pack(side=RIGHT, padx=5)
        ttk.Label(outer, textvariable=self.status_var, foreground="#555555").pack(
            anchor="w", pady=(12, 0)
        )

        root.bind("<Left>", self.go_previous)
        root.bind("<Right>", self.go_next)
        root.bind("<Return>", self.next_unlabelled)
        root.bind("<KP_Enter>", self.next_unlabelled)
        for value in range(1, 6):
            root.bind(str(value), lambda _event, selected=value: self.set_label(selected))
        self.show_sample()

    def show_sample(self) -> None:
        sample = self.samples[self.index]
        current_label = self.labels.get(sample.key)
        self.location_var.set(
            f"{sample.arxiv_id}  •  paragraph {self.index + 1} of {len(self.samples)}"
        )
        self.details_var.set(self.details(sample))
        self.section_var.set(sample.section_header or "(Untitled section)")
        self.paragraph_text.configure(state="normal")
        self.paragraph_text.delete("1.0", "end")
        self.paragraph_text.insert("end", sample.paragraph)
        self.paragraph_text.configure(state="disabled")
        for value, button in enumerate(self.label_buttons, start=1):
            button.configure(relief="sunken" if value == current_label else "raised")
        labelled = sum(sample.key in self.labels for sample in self.samples)
        self.progress.configure(maximum=len(self.samples), value=labelled)
        self.progress_var.set(f"{labelled} labelled • {len(self.samples) - labelled} remaining")

    def set_label(self, value: int):
        self.labels[self.samples[self.index].key] = value
        self.save(self.labels)
        next_index = self._find_unlabelled(start=self.index, wrap=True)
        if next_index is None:
            self.status_var.set("All paragraphs are labelled. You can still revise labels.")
        else:
            self.index = next_index
        self.show_sample()
        return "break"

    def _find_unlabelled(self, start: int, wrap: bool) -> int | None:
        indices = list(range(start + 1, len(self.samples)))
        if wrap and start > 0:
            indices += list(range(0, start))
        return next((index for index in indices if self.samples[index].key not in self.labels), None)

    def next_unlabelled(self, _event=None):
        next_index = self._find_unlabelled(start=self.index, wrap=True)
        if next_index is not None:
            self.index = next_index
            self.show_sample()
        return "break"

    def go_previous(self, _event=None):
        self.index = max(0, self.index - 1)
        self.show_sample()
        return "break"

    def go_next(self, _event=None):
        self.index = min(len(self.samples) - 1, self.index + 1)
        self.show_sample()
        return "break"
