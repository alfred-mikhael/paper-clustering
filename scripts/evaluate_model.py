#!/usr/bin/env python3
"""Evaluate SciBERT or Gemma paragraph classifications."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_clustering.compute_metrics import compute_metrics
from paper_clustering.data_models import ArxivSection, ModelMetrics
from paper_clustering.label.slm_labels import SLMWeakLabelGen
from paper_clustering.technique_classifier import SciBERTClassifier
from paper_clustering.train.load_data import load_labelled_paragraphs

CLASS_LABELS = "ABCDE"
DEFAULT_TEST_DATASET = Path("test_dataset.tsv")


def _choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_gemma_probabilities(
    dataset_path: str | Path,
    *,
    server_url: str | None = None,
    request_timeout: float = 300.0,
    cache_dir: str | Path | None = None,
) -> torch.Tensor:
    """Label test papers with Gemma and return probabilities in TSV order."""
    with Path(dataset_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "arxiv_id",
            "section_index",
            "paragraph_index",
            "section_header",
            "paragraph",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{dataset_path} must contain paragraph identity columns")
        rows = [row for row in reader if row["paragraph"].strip()]

    keys = []
    rows_by_paper = defaultdict(list)
    for line_number, row in enumerate(rows, start=2):
        try:
            key = (
                row["arxiv_id"].strip(),
                int(row["section_index"]),
                int(row["paragraph_index"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid paragraph identity on line {line_number}"
            ) from exc
        keys.append(key)
        rows_by_paper[key[0]].append((key, row))

    labeler = SLMWeakLabelGen(
        server_url=server_url,
        request_timeout=request_timeout,
        cache_dir=cache_dir,
    )
    probabilities_by_key = {}
    try:
        for paper_id, paper_rows in rows_by_paper.items():
            rows_by_section = defaultdict(list)
            for key, row in paper_rows:
                rows_by_section[key[1]].append((key, row))

            sections = []
            ordered_keys = []
            for section_index in sorted(rows_by_section):
                section_rows = sorted(
                    rows_by_section[section_index], key=lambda item: item[0][2]
                )
                headers = {row["section_header"].strip() for _, row in section_rows}
                if len(headers) != 1:
                    raise ValueError(
                        f"Inconsistent headers for section {section_index} "
                        f"of {paper_id}"
                    )
                sections.append(
                    ArxivSection(
                        arxiv_id=paper_id,
                        section_index=section_index,
                        section_header=headers.pop(),
                        major_section_header="",
                        text=tuple(row["paragraph"].strip() for _, row in section_rows),
                    )
                )
                ordered_keys.extend(key for key, _ in section_rows)

            paper = SimpleNamespace(sections=tuple(sections))
            paper_probabilities = labeler.label_paper(paper)
            probabilities_by_key.update(zip(ordered_keys, paper_probabilities))
    finally:
        labeler.session.close()

    return torch.stack([probabilities_by_key[key] for key in keys])


def print_de_false_negatives(
    paper_ids: list[str],
    passages: list[str],
    truth: torch.Tensor,
    logits: torch.Tensor,
    k: int,
) -> None:
    """Print up to ``k`` confident A-C predictions whose true label is D/E."""
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    false_negatives = [
        index
        for index in range(len(passages))
        if truth[index].item() >= 3 and predictions[index].item() < 3
    ]
    false_negatives.sort(
        key=lambda index: probabilities[index, predictions[index]].item(),
        reverse=True,
    )

    selected = false_negatives[:k]
    print(
        f"\nD/E paragraphs classified as A-C "
        f"(showing {len(selected)} of {len(false_negatives)}):"
    )
    for sample_number, index in enumerate(selected, start=1):
        predicted = predictions[index].item()
        confidence = probabilities[index, predicted].item()
        print(
            f"\n{sample_number}. Paper: {paper_ids[index]} | "
            f"truth: {CLASS_LABELS[truth[index].item()]} | "
            f"predicted: {CLASS_LABELS[predicted]} ({confidence:.1%})"
        )
        print(passages[index])


def print_de_false_positives(
    paper_ids: list[str],
    passages: list[str],
    truth: torch.Tensor,
    logits: torch.Tensor,
    k: int,
) -> None:
    """Print up to ``k`` confident D/E predictions whose true label is A-C."""
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    false_positives = [
        index
        for index in range(len(passages))
        if truth[index].item() < 3 and predictions[index].item() >= 3
    ]
    false_positives.sort(
        key=lambda index: probabilities[index, predictions[index]].item(),
        reverse=True,
    )

    selected = false_positives[:k]
    print(
        f"\nA-C paragraphs classified as D/E "
        f"(showing {len(selected)} of {len(false_positives)}):"
    )
    for sample_number, index in enumerate(selected, start=1):
        predicted = predictions[index].item()
        confidence = probabilities[index, predicted].item()
        print(
            f"\n{sample_number}. Paper: {paper_ids[index]} | "
            f"truth: {CLASS_LABELS[truth[index].item()]} | "
            f"predicted: {CLASS_LABELS[predicted]} ({confidence:.1%})"
        )
        print(passages[index])


def evaluate(
    dataset_path: str | Path = DEFAULT_TEST_DATASET,
    checkpoint_path: str | Path = SciBERTClassifier.DEFAULT_WEIGHTS_PATH,
    *,
    model_type: str = "scibert",
    batch_size: int = 32,
    device: str | None = None,
    k: int = 10,
    server_url: str | None = None,
    request_timeout: float = 300.0,
    cache_dir: str | Path | None = None,
) -> ModelMetrics:
    """Evaluate SciBERT or Gemma, print metrics, and return them."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if k < 0:
        raise ValueError("k must be nonnegative")

    paper_ids, passages, label_distributions = load_labelled_paragraphs(dataset_path)
    if any(not paper_id for paper_id in paper_ids):
        raise ValueError(f"{dataset_path} must contain an arxiv_id for every paragraph")
    truth = torch.tensor(
        [max(range(5), key=distribution.__getitem__) for distribution in label_distributions],
        dtype=torch.long,
    )
    if model_type == "gemma":
        probabilities = predict_gemma_probabilities(
            dataset_path,
            server_url=server_url,
            request_timeout=request_timeout,
            cache_dir=cache_dir,
        )
        logits = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    elif model_type == "scibert":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {checkpoint_path}"
            )
        selected_device = _choose_device(device)
        model = SciBERTClassifier(weights_path=checkpoint_path).to(selected_device)
        logits = model.predict_logits(passages, batch_size)
    else:
        raise ValueError("model_type must be 'scibert' or 'gemma'")
    metrics = compute_metrics(paper_ids, logits, truth)

    print(
        f"Evaluated {model_type} on {len(passages)} paragraphs "
        f"from {len(set(paper_ids))} papers"
    )
    print(f"Mean Precision@{metrics.k}: {metrics.mean_precision_at_k:.4f}")
    print(f"Brier score: {metrics.brier_score:.4f}")
    print(f"Binary cross-entropy: {metrics.bce:.4f}")
    print(f"Macro F1: {metrics.macro_f1:.4f}")
    for label, accuracy, recall, precision, f1 in zip(
        CLASS_LABELS,
        metrics.classwise_accuracy,
        metrics.classwise_recall,
        metrics.classwise_precision,
        metrics.classwise_f1,
    ):
        print(
            f"{label}: accuracy={accuracy:.4f}, recall={recall:.4f}, "
            f"precision={precision:.4f}, F1={f1:.4f}"
        )
    if k:
        print_de_false_negatives(paper_ids, passages, truth, logits, k)
        print_de_false_positives(paper_ids, passages, truth, logits, k)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_TEST_DATASET)
    parser.add_argument(
        "--model", choices=("scibert", "gemma"), default="scibert"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=SciBERTClassifier.DEFAULT_WEIGHTS_PATH
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="for example: cuda, cpu, or mps")
    parser.add_argument(
        "-k",
        "--error-samples",
        type=int,
        default=10,
        help="number of true D/E paragraphs misclassified as A-C to print",
    )
    parser.add_argument("--server-url", help="llama.cpp URL for Gemma evaluation")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--cache-dir", type=Path, help="Gemma label cache directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate(
        args.dataset,
        args.checkpoint,
        model_type=args.model,
        batch_size=args.batch_size,
        device=args.device,
        k=args.error_samples,
        server_url=args.server_url,
        request_timeout=args.request_timeout,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
