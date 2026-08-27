"""Fine-tune :class:`SciBERTClassifier` on labelled paragraphs.

Run from the repository root with::

    python -m paper_clustering.train.train_classifier paragraph_labels.tsv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..technique_classifier import SciBERTClassifier

LABEL_FIELDS = tuple(f"probability_{label}" for label in "ABCDE")


def _normalize_distribution(values: Sequence[float]) -> list[float]:
    distribution = [float(value) for value in values]
    if (
        len(distribution) != 5
        or any(not math.isfinite(value) or value < 0 for value in distribution)
        or sum(distribution) <= 0
    ):
        raise ValueError("labels must be length-5 probability distributions")
    total = sum(distribution)
    return [value / total for value in distribution]


def load_labelled_paragraphs(
    path: str | Path,
) -> tuple[list[str], list[list[float]]]:
    """Load human one-hot labels or Gemma probability distributions."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        if not {"section_header", "paragraph", "label"}.issubset(fields):
            raise ValueError(f"{path} must contain paragraph and label columns")
        probability_fields = fields.intersection(LABEL_FIELDS)
        if probability_fields and probability_fields != set(LABEL_FIELDS):
            raise ValueError(f"{path} must contain all five probability_A-E columns")
        has_probabilities = bool(probability_fields)

        paragraphs = []
        labels = []
        for line_number, row in enumerate(reader, start=2):
            section = row["section_header"].strip()
            paragraph = row["paragraph"].strip()
            if not paragraph:
                continue
            try:
                if has_probabilities:
                    distribution = _normalize_distribution(
                        [row[field] for field in LABEL_FIELDS]
                    )
                else:
                    human_label = float(row["label"])
                    if not human_label.is_integer() or not 1 <= human_label <= 5:
                        raise ValueError
                    distribution = [0.0] * 5
                    distribution[int(human_label) - 1] = 1.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label on line {line_number}") from exc
            paragraphs.append(f"Section: {section} Text: {paragraph}")
            labels.append(distribution)

    if not paragraphs:
        raise ValueError(f"No labelled paragraphs found in {path}")
    return paragraphs, labels


def print_dataset_statistics(
    paragraphs: Sequence[str],
    labels: Sequence[Sequence[float]],
    tokenizer,
) -> None:
    """Print token-length and label-distribution summary statistics."""
    tokenized = tokenizer(
        list(paragraphs),
        add_special_tokens=True,
        padding=False,
        truncation=False,
        verbose=False,
    )
    lengths = [len(input_ids) for input_ids in tokenized["input_ids"]]
    expected_scores = [
        sum(score * probability for score, probability in enumerate(label, start=1))
        for label in labels
    ]

    average_length = sum(lengths) / len(lengths)
    over_limit = 100 * sum(length > 512 for length in lengths) / len(lengths)
    at_least_c = 100 * sum(score >= 3 for score in expected_scores) / len(labels)
    print("Dataset statistics:")
    print(f"  {len(paragraphs)} training paragraphs")
    print(f"  Average paragraph length: {average_length:.1f} tokens")
    print(f"  Paragraphs exceeding 512 tokens: {over_limit:.1f}%")
    print(f"  Labels with expected score >= C: {at_least_c:.1f}%")


def _collate(tokenizer, max_length: int):
    def collate(
        samples: list[tuple[str, list[float]]],
    ) -> dict[str, torch.Tensor]:
        paragraphs, labels = zip(*samples)
        batch = tokenizer(
            list(paragraphs),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    return collate


def _choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_loss(
    model: SciBERTClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            logits = model(**{key: value.to(device) for key, value in batch.items()})
            losses.append(loss_function(logits, labels).item())
    return sum(losses) / len(losses)


def train_classifier(
    paragraphs: Sequence[str],
    labels: Sequence[Sequence[float]],
    *,
    checkpoint_path: str | Path = SciBERTClassifier.DEFAULT_WEIGHTS_PATH,
    model_name: str = SciBERTClassifier.DEFAULT_MODEL_NAME,
    mlp_hidden_size: int = 256,
    dropout: float = 0.1,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device: str | None = None,
) -> SciBERTClassifier:
    """Train the classifier with five-way cross-entropy."""
    if len(paragraphs) != len(labels) or not paragraphs:
        raise ValueError("paragraphs and labels must be nonempty and equally sized")
    if epochs < 1 or batch_size < 1 or max_length < 1 or learning_rate <= 0:
        raise ValueError("training parameters must be positive")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    normalized_labels = [_normalize_distribution(label) for label in labels]
    torch.manual_seed(seed)
    selected_device = _choose_device(device)
    checkpoint_path = Path(checkpoint_path)
    model = SciBERTClassifier(
        model_name=model_name,
        mlp_hidden_size=mlp_hidden_size,
        dropout=dropout,
        weights_path=checkpoint_path,
    ).to(selected_device)
    model.scibert.gradient_checkpointing_enable()
    print_dataset_statistics(paragraphs, normalized_labels, model.tokenizer)

    dataset = list(zip(paragraphs, normalized_labels))
    validation_size = int(len(dataset) * validation_fraction)
    training_data, validation_data = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    collate = _collate(model.tokenizer, max_length)
    training_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    validation_loader = (
        DataLoader(validation_data, batch_size=batch_size, collate_fn=collate)
        if validation_size
        else None
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    if validation_loader:
        best_validation_loss = _mean_loss(
            model, validation_loader, loss_function, selected_device
        )
        print(f"baseline validation_loss={best_validation_loss:.6f}")

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(training_loader):
            labels_for_batch = batch.pop("labels").to(selected_device)
            logits = model(
                **{key: value.to(selected_device) for key, value in batch.items()}
            )
            loss = loss_function(logits, labels_for_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_training_loss = sum(losses) / len(losses)
        message = f"epoch {epoch + 1}/{epochs}: train_loss={mean_training_loss:.6f}"
        if validation_loader:
            validation_loss = _mean_loss(
                model, validation_loader, loss_function, selected_device
            )
            message += f", validation_loss={validation_loss:.6f}"
            if validation_loss > best_validation_loss:
                print(message)
                print("Early stopping: validation loss increased; checkpoint unchanged")
                break
            best_validation_loss = validation_loss
        print(message)
        torch.save(model.state_dict(), checkpoint_path)

    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune SciBERT on paragraph labels"
    )
    parser.add_argument("labels", type=Path, help="TSV produced by label_sentences.py")
    parser.add_argument(
        "--checkpoint", type=Path, default=SciBERTClassifier.DEFAULT_WEIGHTS_PATH
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="training batch size",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="for example: cuda, cpu, or mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paragraphs, labels = load_labelled_paragraphs(args.labels)
    train_classifier(
        paragraphs,
        labels,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
