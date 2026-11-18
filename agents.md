# Project context for coding agents

## Purpose and direction

This project organizes arXiv papers in mathematics and computer science. Its existing backend extracts source text, identifies technique-bearing passages with a SciBERT classifier, embeds papers and passages, generates topic clusters and map coordinates, and serves related-paper retrieval.

The intended direction is an improved topic visualization plus a catalogue of documented connections between fields. The visualization product already exists in a separate repository; its frontend is not missing project functionality to recreate here. This repository provides its data and backend infrastructure.

In the new scope, "reduction" includes formal reductions, reformulations, and applications of ideas uncommon in the receiving field, including imported theorems and methods. Historical novelty is not required: a familiar result can be new to a user. Initial experiments target explicit textual evidence of these transfers, accepting that unstated origins limit coverage.

Purpose-based retrieval is conditional on extracting faithful, useful explanations of why an idea was applied. Successful search finds connections that helped solve similar problems; it does not diagnose the user's research obstacles or prescribe their next step. See [restructure.md](restructure.md) for the detailed plan and open hypotheses, and [readme.md](readme.md) for current commands and operational constraints. Planned capabilities should not be described as implemented.

## Repository structure

| Path | Responsibility |
| --- | --- |
| `paper_clustering/pipeline.py` | CLI orchestration for selection, downloading, embedding, coordinates, clustering, and uploads. |
| `paper_clustering/extract/` | Metadata and TeX extraction, technique selection, and theorem/proof statement extraction. |
| `paper_clustering/data_models.py`, `interfaces.py` | Shared records and interfaces. |
| `paper_clustering/local_storage.py`, `postgres.py` | Resumable local artifacts and PostgreSQL operations. |
| `paper_clustering/embedding.py`, `technique_classifier.py` | Embedding inference and existing passage classification. |
| `paper_clustering/cluster.py`, `generate_coords.py` | Hierarchical clustering, labels, and visualization coordinates. |
| `paper_clustering/query.py`, `reranker.py`, `api.py` | Passage retrieval, reranking, and FastAPI endpoints. |
| `paper_clustering/contextualize.py`, `llama.py` | Statement contextualization and llama.cpp access. |
| `paper_clustering/label/`, `train/`, `compute_metrics.py` | Label generation, training/active learning, and evaluation metrics. |
| `scripts/` | Data preparation, annotation, training, and evaluation commands; see `scripts/README.md`. |
| `notebooks/` | Exploratory experiments, including `technique_clustering.ipynb` and `extract_reductions.ipynb`. |
| `tests/` | Unit tests for extraction, storage, pipeline behavior, clustering, retrieval, training utilities, and service integrations. |
| `data/`, `weights/` | Local datasets, cached sources, embeddings, experiment outputs, and model checkpoints; generally ignored by Git. |
| `pyproject.toml`, `requirements.txt` | Packaging, CLI entry point, and dependencies. |
| `readme.md`, `restructure.md`, `worklog.md` | Current usage, intended scope/experiments, and historical work notes. |

The current reduction notebook is an exploratory paragraph-by-paragraph teacher extractor, not a validated production connection index or purpose-search system. The existing search API retrieves technique passages. Database schema migrations/bootstrap are not supplied here; consult the README and storage contracts before assuming a deployed schema.

## Intended implementation sequence

1. Define a connection annotation rubric and update the teacher prompt with borderline cases and hard negatives.
2. Build a human-reviewed development set and a separate, larger final benchmark reflecting realistic prevalence. Distinguish missing context, extraction errors, and actual negatives.
3. Compare paragraph, abstract/introduction, and whole-paper teacher context on the same examples. Check whether identified passages are independently classifiable before choosing a student model.
4. Evaluate purpose extraction: objective, stated obstacle, and the imported idea's role, supported by source text. Allow missing information and avoid filling gaps from outside knowledge.
5. Test whether these purposes improve retrieval over existing passage search on a manually checked catalogue. This is the main uncertainty for the proposed search experience.
6. Select and evaluate an efficient complete pipeline only after establishing context requirements and usefulness. A candidate classifier and a structured-purpose extractor are distinct stages; account for both in cost and quality measurements.
7. Scale indexing after held-out evaluation, preserving provenance and model/prompt versions and grouping duplicate descriptions for presentation.
8. Improve the existing map's labels, filtering, and corpus coverage in coordination with its separate repository. Keep topic positions separate from cross-field connection edges; evaluate topic embedding fine-tuning on map quality.

Also run a bounded technique-retrieval experiment with manually selected passages and known matches: weak clusters alone do not establish weak ranked retrieval. The ambiguity of general technique similarity remains a separate reason for the scope change.

## Working conventions

- Make only requested changes; clarify ambiguous scope or necessary changes to additional files before editing them.
- Use `apply_patch` for code changes, preserve unrelated work, and prefer simple, reusable functions with clear responsibilities.
- Reuse existing extraction, storage, embedding, and retrieval components where appropriate rather than duplicating them.
- Validate changes with relevant checks. The documented unit-test command is `python -m unittest discover -s tests`; some tests skip when optional dependencies are unavailable.
- Do not treat the roadmap as authorization to implement all its steps. Keep experiments, validated findings, and production behavior distinct.
