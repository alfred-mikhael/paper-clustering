"""Deprecated: an attempt to train a multiclass regression model to generate pseudolabels from the Gemma probability distribution and the regex features.
It was not successful, as it consistently underranked paragraphs labelled D/E. I'm sure this is partially due to very imbalalnced classes, but I do not
have the time to label 10 more papers to get a larger number of D/E paragraphs."""

import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paper_clustering.extract_text import get_paper
from paper_clustering.train.label_management.regex_labels import (
    FEATURE_NAMES,
    label_paper as regex_features,
)
from paper_clustering.train.label_management.slm_labels import LABELS, SLMWeakLabelGen

ARXIV_IDS = [
    "0910.1649",
    "cs/9910010",
    "1006.1300",
    "2608.24866v1",
    "2608.21594v1",
]
GROUND_TRUTH = Path("weak_label_calibration.tsv")
WEIGHTS = Path("weights/regression.pt")
CONFUSION_MATRIX = Path("weights/regression_confusion_matrix.png")
C_VALUES = (1.0, 0.5, 0.1)
GEMMA_FEATURE_LABELS = ("C", "D", "E")
GEMMA_FEATURE_INDICES = tuple(LABELS.index(label) for label in GEMMA_FEATURE_LABELS)
GEMMA_FEATURE_NAMES = tuple(
    f"gemma_probability_{label}" for label in GEMMA_FEATURE_LABELS
)
NUM_CLASSES = len(LABELS)
CLASS_INDICES = np.arange(NUM_CLASSES)
CLASS_WEIGHTS = {0: 1.5, 1: 10.0, 2: 30.0, 3: 60.0, 4: 80.0}


@dataclass(frozen=True)
class ParagraphRecord:
    arxiv_id: str
    section_index: int
    paragraph_index: int
    section_header: str
    text: str


def read_targets(path: Path, ids: set[str]) -> dict[tuple[str, int, int], np.ndarray]:
    """Read human ratings 1--5 as one-hot vectors ordered like ``LABELS``."""
    with path.open(encoding="utf-8", newline="") as stream:
        targets = {}
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["arxiv_id"] not in ids:
                continue
            rating = int(row["label"])
            if not 1 <= rating <= NUM_CLASSES:
                raise ValueError(
                    f"Human label must be between 1 and {NUM_CLASSES}, got {rating}"
                )
            key = (
                row["arxiv_id"],
                int(row["section_index"]),
                int(row["paragraph_index"]),
            )
            targets[key] = np.eye(NUM_CLASSES, dtype=np.float64)[rating - 1]
        return targets


def collect_data(ids: list[str], targets: dict[tuple[str, int, int], np.ndarray]):
    """Collect selected Gemma and regex inputs for labelled paragraphs."""
    inputs, labels, groups, paragraphs, gemma_scores = [], [], [], [], []
    with requests.Session() as session:
        slm = SLMWeakLabelGen()
        for arxiv_id in ids:
            sections = get_paper(session, arxiv_id)
            slm_labels = slm.label_paper(sections)
            features = regex_features(sections)
            result_index = 0
            for section in sections:
                for paragraph_index in range(len(section.text)):
                    target = targets.get(
                        (arxiv_id, section.section_index, paragraph_index)
                    )
                    slm_label = slm_labels[result_index]
                    feature = features[result_index]
                    result_index += 1
                    if target is None:
                        continue
                    probabilities = [float(probability) for probability in slm_label]
                    inputs.append(
                        [
                            *(probabilities[index] for index in GEMMA_FEATURE_INDICES),
                            *feature,
                        ]
                    )
                    gemma_scores.append(probabilities)
                    labels.append(target)
                    groups.append(arxiv_id)
                    paragraphs.append(
                        ParagraphRecord(
                            arxiv_id=arxiv_id,
                            section_index=section.section_index,
                            paragraph_index=paragraph_index,
                            section_header=section.section_header,
                            text=section.text[paragraph_index],
                        )
                    )
    if not inputs:
        raise ValueError("No matching labelled paragraphs were found")
    return (
        np.asarray(inputs),
        np.stack(labels),
        np.asarray(groups),
        paragraphs,
        np.asarray(gemma_scores),
    )


def print_label_split(name: str, values: np.ndarray) -> None:
    counts = values.sum(axis=0).astype(int)
    split = ", ".join(
        f"{label}: {count} ({count / len(values):.1%})"
        for label, count in zip(LABELS, counts)
    )
    print(f"{name}: {len(values)} paragraphs; {split}")


def fit_model(inputs: np.ndarray, targets: np.ndarray, c: float):
    """Fit multinomial logistic regression after standardizing all inputs."""
    if targets.ndim != 2 or targets.shape[1] != NUM_CLASSES:
        raise ValueError(f"targets must have shape (n_samples, {NUM_CLASSES})")

    classifier = LogisticRegression(
        max_iter=1000,
        C=c,
        solver="lbfgs",
        class_weight=CLASS_WEIGHTS,
    )
    model = Pipeline(
        [
            ("scale_features", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    # LogisticRegression expects class indices even though targets are kept as
    # one-hot vectors throughout data loading and evaluation.
    model.fit(inputs, targets.argmax(axis=1))
    return model


def metric_values(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Calculate multiclass metrics from A--E class probabilities."""
    predictions = probabilities.argmax(axis=1)
    truth = targets.argmax(axis=1)
    sample_weights = np.array([CLASS_WEIGHTS[int(label)] for label in truth])
    return {
        "accuracy": accuracy_score(truth, predictions),
        "macro recall": recall_score(
            truth, predictions, labels=CLASS_INDICES, average="macro", zero_division=0
        ),
        "macro F1": f1_score(
            truth, predictions, labels=CLASS_INDICES, average="macro", zero_division=0
        ),
        "categorical cross-entropy": log_loss(
            truth, probabilities, labels=CLASS_INDICES
        ),
        "D/E-weighted cross-entropy": log_loss(
            truth,
            probabilities,
            labels=CLASS_INDICES,
            sample_weight=sample_weights,
        ),
        "Brier score": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "mean absolute label error": float(np.mean(np.abs(predictions - truth))),
    }


def print_summary(name: str, results: list[dict[str, float]]) -> None:
    """Print macro averages: every held-out paper has equal weight."""
    print(f"\n{name} cross-validation average across {len(results)} papers:")
    for metric in results[0]:
        values = np.array([result[metric] for result in results])
        print(f"{metric}: {values.mean():.4f} ± {values.std():.4f}")


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    """Save cross-validated class counts with true labels on the y-axis."""
    path.parent.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(matrix, display_labels=LABELS)
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Logistic regression: out-of-fold predictions")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(matrix)
    print(f"Saved confusion matrix to {path}")


def print_targeted_errors(
    paragraphs: list[ParagraphRecord],
    targets: np.ndarray,
    predictions: np.ndarray,
    gemma_scores: np.ndarray,
) -> None:
    """Print E false negatives and A--C false positives from cross-validation."""
    truth = targets.argmax(axis=1)
    d_index = LABELS.index("D")
    e_index = LABELS.index("E")
    error_groups = (
        (
            "True E paragraphs predicted below D",
            (truth == e_index) & (predictions < d_index),
            True,
        ),
        (
            "True A/B/C paragraphs predicted as D or E",
            (truth < d_index) & (predictions >= d_index),
            False,
        ),
    )

    for heading, mask, show_gemma_scores in error_groups:
        indices = np.flatnonzero(mask)
        print(f"\n{heading} ({len(indices)}):")
        if not len(indices):
            print("None")
            continue
        for index in indices:
            paragraph = paragraphs[index]
            print(
                f"\n[{paragraph.arxiv_id}; section {paragraph.section_index} "
                f"({paragraph.section_header}); paragraph {paragraph.paragraph_index}]"
            )
            print(
                f"Human: {LABELS[truth[index]]}; "
                f"predicted: {LABELS[predictions[index]]}"
            )
            if show_gemma_scores:
                scores = ", ".join(
                    f"{label}={score:.4f}"
                    for label, score in zip(LABELS, gemma_scores[index])
                )
                gemma_prediction = LABELS[int(gemma_scores[index].argmax())]
                print(f"Gemma scores: {scores}; highest={gemma_prediction}")
            print(paragraph.text)


def cross_validate(
    inputs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    c: float,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """Evaluate one hyperparameter choice, holding out one full paper per fold."""
    results = []
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    out_of_fold_predictions = np.full(len(labels), -1, dtype=np.int64)
    for train_indices, test_indices in LeaveOneGroupOut().split(inputs, labels, groups):
        train_inputs, train_labels = inputs[train_indices], labels[train_indices]
        model = fit_model(train_inputs, train_labels, c)
        test_probabilities = model.predict_proba(inputs[test_indices])
        test_predictions = test_probabilities.argmax(axis=1)
        out_of_fold_predictions[test_indices] = test_predictions
        results.append(metric_values(test_probabilities, labels[test_indices]))
        matrix += confusion_matrix(
            labels[test_indices].argmax(axis=1),
            test_predictions,
            labels=CLASS_INDICES,
        )
    return results, matrix, out_of_fold_predictions


def main():
    targets = read_targets(GROUND_TRUTH, set(ARXIV_IDS))
    inputs, labels, groups, paragraphs, gemma_scores = collect_data(ARXIV_IDS, targets)
    if set(groups) != set(ARXIV_IDS):
        raise ValueError("Every arXiv ID must have at least one labelled paragraph")

    # Gemma needs no fitting; evaluate its raw class probabilities on each paper.
    gemma_results = [
        metric_values(gemma_scores[test_indices], labels[test_indices])
        for _, test_indices in LeaveOneGroupOut().split(inputs, labels, groups)
    ]

    candidates = []
    selection_metric = "D/E-weighted cross-entropy"
    for c in C_VALUES:
        results, matrix, predictions = cross_validate(inputs, labels, groups, c)
        candidates.append(
            (
                np.mean([result[selection_metric] for result in results]),
                c,
                results,
                matrix,
                predictions,
            )
        )

    _, best_c, best_results, best_matrix, best_predictions = min(
        candidates, key=lambda item: item[0]
    )
    print("Best configuration (selected by mean " + selection_metric + "):")
    print(f"C: {best_c}")
    print(f"class weights: {dict(zip(LABELS, CLASS_WEIGHTS.values()))}")
    print(f"Gemma features: {list(GEMMA_FEATURE_NAMES)}")
    print(f"regex features: {list(FEATURE_NAMES)}")
    print_summary("Best logistic regression", best_results)
    print_summary("Gemma only", gemma_results)
    save_confusion_matrix(best_matrix, CONFUSION_MATRIX)
    print_targeted_errors(
        paragraphs,
        labels,
        best_predictions,
        gemma_scores,
    )

    # Save a final model fitted to every labelled paper with the selected setup.
    final_inputs = inputs
    final_labels = labels
    final_model = fit_model(final_inputs, final_labels, best_c)
    WEIGHTS.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_names": [*GEMMA_FEATURE_NAMES, *FEATURE_NAMES],
            "class_labels": list(LABELS),
            "class_weights": dict(zip(LABELS, CLASS_WEIGHTS.values())),
            "C": best_c,
            "selection_metric": selection_metric,
        },
        WEIGHTS,
    )


if __name__ == "__main__":
    main()
