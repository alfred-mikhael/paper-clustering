"""Metrics for paragraph-technique classifiers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.nn import functional as F

from .data_models import ModelMetrics


def mean_precision_at_k(
    eval_ids: Sequence[str],
    truth: torch.Tensor,
    probabilities: torch.Tensor,
    k: int,
) -> float:
    """Return mean per-paper P@k, treating D and E as relevant."""
    if k < 1:
        raise ValueError("k must be positive")

    indices_by_paper: dict[str, list[int]] = defaultdict(list)
    for index, paper_id in enumerate(eval_ids):
        indices_by_paper[paper_id].append(index)

    relevance_scores = probabilities[:, 3:].sum(dim=1)
    paper_precisions = []
    for indices in indices_by_paper.values():
        top_indices = sorted(
            indices,
            key=lambda index: relevance_scores[index].item(),
            reverse=True,
        )[:k]
        relevant = sum(truth[index].item() >= 3 for index in top_indices)
        paper_precisions.append(relevant / len(top_indices))
    return sum(paper_precisions) / len(paper_precisions)


def compute_metrics(
    eval_ids: Sequence[str], logits: torch.Tensor, truth: torch.Tensor, *, k: int = 10
) -> ModelMetrics:
    """Compute multiclass calibration and retrieval metrics from model logits."""
    if k < 1:
        raise ValueError("k must be positive")
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("logits must have shape (number of samples, 5)")
    if truth.ndim != 1 or truth.shape[0] != logits.shape[0]:
        raise ValueError("truth must contain one label for every logit row")
    if len(eval_ids) != len(truth):
        raise ValueError("eval_ids must contain one ID for every label")
    if not len(truth):
        raise ValueError("cannot compute metrics for an empty evaluation set")
    if not torch.all((0 <= truth) & (truth < 5)):
        raise ValueError("truth labels must be in the range 0 through 4")

    logits = logits.detach().cpu()
    truth = truth.detach().cpu().long()
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    targets = F.one_hot(truth, num_classes=5).to(dtype=probabilities.dtype)

    truth_values = truth.numpy()
    prediction_values = predictions.numpy()
    precision, recall, classwise_f1, _ = precision_recall_fscore_support(
        truth_values,
        prediction_values,
        labels=range(5),
        zero_division=0,
    )
    classwise_accuracy = tuple(
        float(((predictions == label) == (truth == label)).float().mean().item())
        for label in range(5)
    )

    return ModelMetrics(
        k=k,
        eval_ids=list(eval_ids),
        mean_precision_at_k=mean_precision_at_k(eval_ids, truth, probabilities, k),
        brier_score=float(((probabilities - targets).square().sum(dim=1)).mean().item()),
        bce=float(F.binary_cross_entropy(probabilities, targets).item()),
        macro_f1=float(
            f1_score(
                truth_values,
                prediction_values,
                labels=range(5),
                average="macro",
                zero_division=0,
            )
        ),
        classwise_accuracy=classwise_accuracy,
        classwise_recall=tuple(float(value) for value in recall),
        classwise_precision=tuple(float(value) for value in precision),
        classwise_f1=tuple(float(value) for value in classwise_f1),
    )
