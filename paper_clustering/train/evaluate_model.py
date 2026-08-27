"""Evaluate a trained SciBERT paragraph classifier."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score
from torch.nn import functional as F
from tqdm import tqdm

from ..technique_classifier import SciBERTClassifier

CLASS_LABELS = "ABCDE"
DEFAULT_TEST_DATASET = Path("test_dataset.tsv")


def load_test_dataset(
    path: str | Path,
) -> tuple[list[str], list[str], torch.Tensor]:
    """Load paper IDs, model inputs, and zero-based A-E labels from a TSV."""
    paper_ids = []
    passages = []
    labels = []

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"arxiv_id", "section_header", "paragraph", "label"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain {', '.join(sorted(required))}")

        for line_number, row in enumerate(reader, start=2):
            paragraph = row["paragraph"].strip()
            if not paragraph:
                continue
            try:
                label = float(row["label"])
                if not label.is_integer() or not 1 <= label <= 5:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label on line {line_number}") from exc

            paper_ids.append(row["arxiv_id"].strip())
            passages.append(
                f"Section: {row['section_header'].strip()} Text: {paragraph}"
            )
            labels.append(int(label) - 1)

    if not passages:
        raise ValueError(f"No labelled paragraphs found in {path}")
    return paper_ids, passages, torch.tensor(labels, dtype=torch.long)


def _choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_logits(
    model: SciBERTClassifier,
    passages: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run batched inference and return CPU logits."""
    logits = []
    model.eval()
    with torch.inference_mode():
        starts = range(0, len(passages), batch_size)
        for start in tqdm(starts, desc="Evaluating"):
            encoded = model.tokenizer(
                passages[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            logits.append(model(**inputs).cpu())
    return torch.cat(logits)


def mean_precision_at_10(
    paper_ids: list[str],
    truth: torch.Tensor,
    probabilities: torch.Tensor,
) -> float:
    """Average per-paper P@10, with human D/E labels treated as relevant."""
    indices_by_paper = defaultdict(list)
    for index, paper_id in enumerate(paper_ids):
        indices_by_paper[paper_id].append(index)

    relevance_scores = probabilities[:, 3:].sum(dim=1)
    paper_precisions = []
    for indices in indices_by_paper.values():
        top_indices = sorted(
            indices, key=lambda index: relevance_scores[index].item(), reverse=True
        )[:10]
        relevant = sum(truth[index].item() >= 3 for index in top_indices)
        paper_precisions.append(relevant / len(top_indices))
    return sum(paper_precisions) / len(paper_precisions)


def calculate_metrics(
    paper_ids: list[str], logits: torch.Tensor, truth: torch.Tensor
) -> dict[str, float]:
    """Calculate multiclass and D/E retrieval metrics."""
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    truth_de = truth >= 3
    de_percentage = (
        100 * (predictions[truth_de] >= 3).float().mean().item()
        if truth_de.any()
        else 0.0
    )
    per_class_f1 = f1_score(
        truth, predictions, labels=range(5), average=None, zero_division=0
    )

    metrics = {
        "Accuracy": accuracy_score(truth, predictions),
        "Macro F1": f1_score(
            truth, predictions, labels=range(5), average="macro", zero_division=0
        ),
        "Macro precision": precision_score(
            truth, predictions, labels=range(5), average="macro", zero_division=0
        ),
        "Cross-entropy": F.cross_entropy(logits, truth).item(),
        "D/E labels classified as D/E (%)": de_percentage,
        "Mean Precision@10": mean_precision_at_10(paper_ids, truth, probabilities),
    }
    metrics.update(
        {
            f"F1 ({label})": float(score)
            for label, score in zip(CLASS_LABELS, per_class_f1)
        }
    )
    return metrics


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
    batch_size: int = 32,
    device: str | None = None,
    k: int = 10,
) -> dict[str, float]:
    """Load the classifier and test data, print metrics, and return them."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if k < 0:
        raise ValueError("k must be nonnegative")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    paper_ids, passages, truth = load_test_dataset(dataset_path)
    selected_device = _choose_device(device)
    model = SciBERTClassifier(weights_path=checkpoint_path).to(selected_device)
    logits = predict_logits(model, passages, batch_size, selected_device)
    metrics = calculate_metrics(paper_ids, logits, truth)

    print(f"Evaluated {len(passages)} paragraphs from {len(set(paper_ids))} papers")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    if k:
        print_de_false_negatives(paper_ids, passages, truth, logits, k)
        print_de_false_positives(paper_ids, passages, truth, logits, k)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_TEST_DATASET)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate(
        args.dataset,
        args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        k=args.error_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
