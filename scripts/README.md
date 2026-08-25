# arXiv paragraph labeller

Start the GUI with one or more arXiv IDs:

```bash
python scripts/label_sentences.py 1706.03762 2301.00001v2
```

For a longer list, put IDs (or arxiv.org URLs) in a text file, separated by
newlines, spaces, or commas. Lines may contain comments after `#`:

```bash
python scripts/label_sentences.py --ids-file paper_ids.txt --output labels.tsv
```

The first run downloads each paper's TeX source and caches the cleaned samples
in `.paragraph_label_cache/`. Pass `--refresh` to download them again. A paper
that cannot be downloaded or has no extractable TeX sections is reported and
skipped without losing the other papers.

One paragraph appears at a time in the reading area using Arial (or a
compatible system fallback). Press `1`–`5` (or click a numbered button)
to label it and advance to the next unlabelled paragraph. Use the left/right
arrow keys to move through every paragraph, including labelled ones. Press Enter
to jump to the next unlabelled paragraph. The progress bar counts labelled and
remaining paragraphs.

Labels are atomically saved after every choice. The TSV includes the arXiv ID,
section and paragraph positions, section header, paragraph text, and label.
Reopening the same output file resumes the session; relabelling a paragraph
updates its row rather than adding a duplicate.

The GUI uses Python's Tkinter module. On Linux, install your distribution's
`python3-tk` package if Tkinter is not already available.
