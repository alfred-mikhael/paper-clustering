"""Print per-class coefficients of the paragraph logistic-regression model."""

from pathlib import Path

import joblib

from paper_clustering.label_management.regex_labels import FEATURE_NAMES

WEIGHTS = Path("weights/regression.pt")


def main() -> None:
    checkpoint = joblib.load(WEIGHTS)
    if isinstance(checkpoint, dict):
        model = checkpoint["model"]
        names = checkpoint.get("feature_names", [])
        class_labels = checkpoint.get("class_labels", [])
    else:
        model = checkpoint
        names = []
        class_labels = []

    if hasattr(model, "named_steps"):
        classifier = model.named_steps["classifier"]
        names = names or ["slm_probability", *FEATURE_NAMES]
    else:
        classifier = model

    if not names:
        names = ["slm_probability"] + [
            f"regex_feature_{i}" for i in range(classifier.coef_.shape[1] - 1)
        ]

    class_labels = class_labels or [str(value) for value in classifier.classes_]
    for class_label, intercept, coefficients in zip(
        class_labels, classifier.intercept_, classifier.coef_
    ):
        print(f"\nClass {class_label}")
        print(f"intercept: {intercept:.6f}")
        for name, coefficient in zip(names, coefficients):
            print(f"{name:35s} {coefficient:+.6f}")


if __name__ == "__main__":
    main()
