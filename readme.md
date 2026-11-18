# Paper clustering and technique search

Tools for organizing arXiv papers in theoretical computer science and combinatorics by research area and proof technique. The project extracts paragraphs from TeX sources, identifies technique-bearing passages with a fine-tuned SciBERT classifier, embeds papers and passages, and retrieves related papers through a FastAPI service.

The current implementation provides a command-line pipeline, resumable local exports, PostgreSQL storage, and training and labelling tools. It also generates UMAP coordinates and labelled clusters for use in a visualization. Scheduled ingestion, a standalone visualization frontend, title/author search endpoints, citation graphs, and paper digests are not implemented here.

## Installation

Requires Python 3.10 or later. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
paper-clustering --help
```

Runtime dependencies are listed in `requirements.txt`. The CLI can also be invoked as `python -m paper_clustering.pipeline`. Models and tokenizers are downloaded from Hugging Face on first use; full technique extraction also requires a local classifier checkpoint. Data and weights under `data/` and `weights/` are ignored by Git and must be supplied or generated separately.

## Pipeline

### 1. Select paper metadata

`select` reads a local ZIP containing exactly one JSON or JSONL metadata member in the arXiv snapshot format. It does not fetch a fresh snapshot automatically.

```bash
paper-clustering select \
  --input-archive data/math_cs_metadata.zip \
  --days-back 30 \
  --categories cs.CC cs.DS math.CO \
  --output-filepath data/papers.jsonl
```

Dates use the first arXiv version and inclusive UTC calendar days. Optional `--keywords` filters titles and abstracts. A paper must match any supplied category and any supplied keyword; omitted filters impose no restriction. The default limit is 10,000 papers, in archive order. An old snapshot may produce no recent matches.

To prepare a smaller snapshot from an existing arXiv metadata ZIP:

```bash
python scripts/filter_math_cs_metadata.py data/archive.zip data/math_cs_metadata.jsonl
python -m zipfile -c data/math_cs_metadata.zip data/math_cs_metadata.jsonl
```

### 2. Download and cache extracted sources

```bash
paper-clustering download \
  --input-filepath data/papers.jsonl \
  --output-path data/sources \
  --resume
```

This downloads TeX sources and saves cleaned sections and metadata as one compressed JSON file per paper. Alternatively, use `--ids-filepath data/ids.txt` with one arXiv ID per line; metadata is then fetched from arXiv. Add `--include-proofs` to retain proof text, and use the same setting when embedding cached sources.

`--resume` reuses saved papers. Papers without usable source paragraphs are also saved, allowing metadata-only embeddings; remove the corresponding cached file to retry its source download.

### 3. Embed papers and techniques

```bash
paper-clustering embed \
  --input-filepath data/papers.jsonl \
  --source-path data/sources \
  --classifier-weights weights/technique_classifier_checkpoint.pt \
  --output-path data/embedded \
  --batch-size 500 \
  --resume
```

The pipeline embeds titles and abstracts as area vectors, scores source paragraphs with SciBERT, selects technique passages, and merges similar passages within each paper. Omitting `--source-path` downloads sources during embedding. `--ids-filepath` can replace `--input-filepath` for the full extraction workflow.

For area embeddings alone, no classifier checkpoint or source download is needed:

```bash
paper-clustering embed \
  --input-filepath data/papers.jsonl \
  --areas-only \
  --output-path data/areas
```

Supply `--output-path`, `--upload`, or both. `--upload` writes to PostgreSQL. Local output uses `batch_0001/`, `batch_0002/`, etc., each containing:

- `embeddings.pt.gz` and `metadata.csv.gz`: area vectors and paper metadata.
- `techniques.pt.gz` and `techniques.csv.gz`: passage vectors and passage metadata.
- `manifest.json`: settings and progress used to validate resumed runs.

`--resume` requires local output and validates saved batches before reuse. Keep inputs and processing settings consistent when resuming. `--device`, `--embedding-batch-size`, and `--technique-batch-size` control inference; `--semchunk` optionally chunks sections before classification. See `paper-clustering embed --help` for all options.

### 4. Generate visualization coordinates

Fit a two-dimensional UMAP reducer and save it for later use:

```bash
paper-clustering generate_coords \
  --input-filepath data/embedded \
  --metadata-filepath data/embedded \
  --save-umap-weights weights/umap.joblib \
  --output-filepath data/coords.csv.gz
```

To transform another batch with the same reducer, replace `--save-umap-weights` with `--umap-weights weights/umap.joblib`. The output columns are `arxiv_id,title,umap_x,umap_y`. Local inputs may be batch directories or a tensor file and matching compressed metadata CSV. Omit both input paths to read area embeddings from PostgreSQL. Fitting and database reads load the full input into memory; transforms of local inputs support batching.

### 5. Cluster and label papers

Clustering uses HDBSCAN on area vectors, with optional UMAP reduction via `--umap-dim`. Cluster labels are generated through a running llama.cpp chat-completions server.

**The cluster command also uploads results to PostgreSQL, even with local input.** Configure the database and labelling server before running:

```bash
paper-clustering cluster \
  --input-filepath data/embedded \
  --metadata-filepath data/embedded \
  --output-path data/clusters \
  --server-url http://127.0.0.1:8080 \
  --min-cluster-size 5 \
  --seed 42
```

Outputs include `memberships.csv.gz`, `clusters.csv.gz`, and a progress manifest. Repeating the command with the same inputs, settings, and output directory resumes saved labelling/upload progress. Omitting both input paths reads papers from the database.

Estimate cluster centers in the visualization from sampled coordinates:

```bash
paper-clustering generate_centroids \
  --memberships-filepath data/clusters/memberships.csv.gz \
  --coordinates-filepath data/coords.csv.gz \
  --output-filepath data/centroids.csv.gz \
  --seed 42
```

## Database and search API

Database operations use PostgreSQL with pgvector. Export `DATABASE_URL`, or configure the standard `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD` environment variables. The pipeline and API do not explicitly load database configuration from `.env`; provide it in the process environment.

The database must already contain the `papers`, `techniques`, `cluster_metadata`, and `clusters` tables needed by the workflows you run, their upsert constraints, and an `ann` function for passage search. This repository does not include a database schema migration or bootstrap script. The expected records and conflict targets are defined in `paper_clustering/pipeline.py` and `paper_clustering/postgres.py`; the search contract is in `paper_clustering/query.py`.

After uploading papers and technique embeddings:

```bash
uvicorn paper_clustering.api:app --host 127.0.0.1 --port 8000
```

- `GET /health`: service health response.
- `GET /query?arxiv_id=1706.03762&k=10`: related papers ranked by passage reranker scores.
- `GET /query?arxiv_id=1706.03762&k=10&rerank=false`: rank using vector similarity.
- `GET /docs`: interactive API documentation.

Search starts from an arXiv ID with stored technique passages and returns matching passage pairs grouped by target paper ID. `k` ranges from 1 to 100. The reported `similarity` remains the vector similarity even when reranking is enabled. Papers without stored technique passages return no matches.

API startup connects to the database and loads the reranker, including when requests use `rerank=false`. Use one Uvicorn worker on a single GPU because each worker loads its own model and connection pool.

## Models and training

| Purpose | Current default |
| --- | --- |
| Technique paragraph classifier | `allenai/scibert_scivocab_uncased` with a five-class MLP head and local fine-tuned weights |
| Area and technique embeddings | `nomic-ai/nomic-embed-text-v2-moe` |
| Search reranker | `Qwen/Qwen3-Reranker-0.6B` |
| Weak paragraph labels | Gemma through llama.cpp; see `scripts/README.md` |
| Cluster labels | Model served by the configured llama.cpp endpoint |

The llama.cpp connector uses `--server-url` when supplied, otherwise `LLAMA_CPP_SERVER_URL`, falling back to `http://127.0.0.1:8080`. This connector loads `.env`.

Create human labels, train a checkpoint, and evaluate it on a separate labelled dataset:

```bash
python scripts/label_sentences.py --ids-file data/ids.txt --output labels.tsv
python scripts/train_classifier.py labels.tsv --checkpoint weights/technique_classifier_checkpoint.pt
python scripts/evaluate_model.py held_out_labels.tsv --checkpoint weights/technique_classifier_checkpoint.pt
```

The paragraph labelling GUI uses Tkinter and saves after each choice. On Linux, Tkinter may require your distribution's `python3-tk` package. Additional scripts generate weak labels, select active-learning samples, and label the selected paragraphs. See [the labelling guide](scripts/README.md) and each script's `--help` for usage.

## Development

Run the unittest suite, optionally with coverage:

```bash
python -m unittest discover -s tests
python -m coverage run -m unittest discover -s tests
python -m coverage report
```

Tests cover metadata and text extraction, classifier data handling, active learning, clustering, local storage, pipeline commands and checkpoints, database operations, search ranking, and HTTP error/retry handling. Some tests skip when optional dependencies are unavailable.

The main code lives in `paper_clustering/`: `extract/` handles metadata and TeX processing, `label/` and `train/` support classifier development, `pipeline.py` implements the CLI, and `api.py` exposes search. `scripts/` contains standalone data preparation and training tools; `notebooks/` contains exploratory work.
