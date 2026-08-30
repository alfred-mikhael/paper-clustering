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

## Label selected active-learning paragraphs

After creating a selection TSV, label only those paragraphs with:

```bash
python scripts/label_active_learning.py selected_paragraphs.tsv
```

The default output is `selected_paragraphs_labels.tsv`. It preserves the
selection strategy and model diagnostics, adds a human `label`, and is saved
after every choice so the session can be resumed.

# Bulk label papers with SLM

Generate weak labels for paragraphs using `paper_clustering.label.SLMWeakLabelGen`, which is a Gemma-4-E4B-it model. 

```bash
python scripts/generate_pretraining_labels.py --input-path pretraining_arxiv_labels.txt --output-path pretraining_labels.tsv
```

Downloads each paper from arxiv, extracts the parargraphs from source, and runs the SLM on them for classification. Saves 
output of paper `<arxiv_id>` in a `.tsv` file in `<arxiv_id>.slm_label_cache`, and will not recompute labels which already exist.  
In order for this to run, there needs to be a `llama.cpp` server running. You should clone `llama.cpp`, compile, and then run 
```bash 
build/bin/llama-server -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_M  -ngl 99 -c 8192 --cache-prompt --no-mmproj
```
In order to start the server. If you want to change the ip:port, you must pass in the server-url when running the script.

# Train the SciBERT Classifier 

Fine-tune the SciBERT classifier, with the entire 
