"""Minimal, resumable training harness for :class:`TechniqueClassifier`.

Run it from the repository root with::

    python -m src.train.train_classifier paragraph_labels.tsv

The TSV format is the one written by ``scripts/label_sentences.py``. Its labels
range from 1 to 5, so this harness maps them linearly to soft binary targets from
0.0 to 1.0. ``BCEWithLogitsLoss`` supports these probabilistic targets directly.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from .technique_classifier import TechniqueClassifier


class TokenizedParagraphDataset(Dataset):
    """Tokenize paragraphs once and expose them to a PyTorch DataLoader."""

    def __init__(self, paragraphs, labels, tokenizer, max_length: int) -> None:
        # Padding is deliberately omitted here. The collate function below pads
        # each batch only to its longest paragraph, avoiding unnecessary SciBERT
        # computation when paragraph lengths vary substantially.
        encoded = tokenizer(
            list(paragraphs),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.encodings = {
            key: value
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        self.labels = [float(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def _make_collate_function(tokenizer):
    """Create a collator that dynamically pads tokenized paragraph batches."""

    def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor(
            [sample["labels"] for sample in samples], dtype=torch.float32
        )
        model_inputs = [
            {key: value for key, value in sample.items() if key != "labels"}
            for sample in samples
        ]
        batch = tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


def load_labelled_paragraphs(path: str | Path) -> tuple[list[str], list[float]]:
    """Load GUI-generated paragraphs and map labels 1–5 onto probabilities."""
    paragraphs: list[str] = []
    labels: list[float] = []

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_fields = {"paragraph", "label"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain tab-separated paragraph and label columns"
            )

        for line_number, row in enumerate(reader, start=2):
            paragraph = row["paragraph"].strip()
            try:
                five_point_label = float(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label on line {line_number}") from exc
            if not 1.0 <= five_point_label <= 5.0:
                raise ValueError(
                    f"Label on line {line_number} is outside the range 1–5"
                )
            if not paragraph:
                continue

            paragraphs.append(paragraph)
            labels.append((five_point_label - 1.0) / 4.0)

    if not paragraphs:
        raise ValueError(f"No labelled paragraphs found in {path}")
    return paragraphs, labels


def _choose_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_training_progress(
    checkpoint_path: Path, optimizer: torch.optim.Optimizer
) -> int:
    """Restore optimizer state and return the number of completed epochs."""
    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        # TechniqueClassifier can load a plain model state dictionary, but such
        # a file has no optimizer or epoch information to resume here.
        return 0
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("completed_epochs", 0))


def _save_checkpoint(
    checkpoint_path: Path,
    model: TechniqueClassifier,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
) -> None:
    """Atomically save model and optimizer state after a completed epoch."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epochs": completed_epochs,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def _evaluate(
    model: TechniqueClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Compute mean validation loss without building gradient tensors."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs)
            loss = loss_function(logits, labels)
            total_loss += loss.item() * labels.numel()
            total_examples += labels.numel()
    return total_loss / total_examples


def train_classifier(
    paragraphs: Sequence[str],
    labels: Sequence[float],
    *,
    checkpoint_path: str | Path = TechniqueClassifier.DEFAULT_WEIGHTS_PATH,
    model_name: str = TechniqueClassifier.DEFAULT_MODEL_NAME,
    mlp_hidden_size: int = 256,
    dropout: float = 0.1,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device: str | None = None,
) -> TechniqueClassifier:
    """Train or resume the classifier and return the resulting model.

    ``labels`` may be hard binary targets (0 or 1) or probabilities between
    zero and one. ``epochs`` is the desired total epoch count: if a checkpoint
    has already completed two epochs and ``epochs=3``, only one more is run.
    """
    if len(paragraphs) != len(labels):
        raise ValueError("paragraphs and labels must have the same length")
    if not paragraphs:
        raise ValueError("at least one paragraph is required")
    if any(not 0.0 <= float(label) <= 1.0 for label in labels):
        raise ValueError("every label must be between 0 and 1")
    if epochs < 1 or batch_size < 1 or max_length < 1:
        raise ValueError("epochs, batch_size, and max_length must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the interval [0, 1)")

    torch.manual_seed(seed)
    selected_device = _choose_device(device)
    checkpoint_path = Path(checkpoint_path)

    # SciBERT requires its matching scientific vocabulary. Tokenization occurs
    # before model construction so malformed inputs fail before loading the much
    # larger encoder weights.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TokenizedParagraphDataset(paragraphs, labels, tokenizer, max_length)

    validation_size = round(len(dataset) * validation_fraction)
    if validation_fraction > 0 and len(dataset) > 1:
        validation_size = min(max(validation_size, 1), len(dataset) - 1)
    else:
        validation_size = 0
    training_size = len(dataset) - validation_size
    split_generator = torch.Generator().manual_seed(seed)
    training_data, validation_data = random_split(
        dataset, [training_size, validation_size], generator=split_generator
    )

    collate = _make_collate_function(tokenizer)
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = (
        DataLoader(validation_data, batch_size=batch_size, collate_fn=collate)
        if validation_size
        else None
    )

    # TechniqueClassifier automatically restores model parameters when this
    # checkpoint exists. Optimizer and epoch state are restored immediately
    # afterwards by _load_training_progress.
    model = TechniqueClassifier(
        model_name=model_name,
        mlp_hidden_size=mlp_hidden_size,
        dropout=dropout,
        weights_path=checkpoint_path,
    ).to(selected_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    completed_epochs = _load_training_progress(checkpoint_path, optimizer)
    loss_function = nn.BCEWithLogitsLoss()

    print(
        f"Training {len(training_data)} paragraphs on {selected_device}; "
        f"validating on {len(validation_data)}"
    )
    if completed_epochs:
        print(f"Resuming after epoch {completed_epochs} from {checkpoint_path}")

    try:
        for epoch in range(completed_epochs, epochs):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for batch in training_loader:
                labels_for_batch = batch.pop("labels").to(selected_device)
                inputs = {
                    key: value.to(selected_device) for key, value in batch.items()
                }

                optimizer.zero_grad(set_to_none=True)
                logits = model(**inputs)
                loss = loss_function(logits, labels_for_batch)
                loss.backward()
                # Clipping protects fine-tuning from an occasional very large
                # update without otherwise changing normal gradients.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * labels_for_batch.numel()
                total_examples += labels_for_batch.numel()

            training_loss = total_loss / total_examples
            message = f"epoch {epoch + 1}/{epochs}: train_loss={training_loss:.6f}"
            if validation_loader is not None:
                validation_loss = _evaluate(
                    model, validation_loader, loss_function, selected_device
                )
                message += f", validation_loss={validation_loss:.6f}"
            print(message)

            completed_epochs = epoch + 1
            _save_checkpoint(checkpoint_path, model, optimizer, completed_epochs)
    except KeyboardInterrupt:
        # Save partial-epoch parameter and optimizer updates. On the next run,
        # that epoch is repeated, but the updates completed so far are retained.
        _save_checkpoint(checkpoint_path, model, optimizer, completed_epochs)
        print(f"Interrupted; saved progress to {checkpoint_path}")
        raise

    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune SciBERT on paragraph labels")
    parser.add_argument("labels", type=Path, help="TSV produced by label_sentences.py")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=TechniqueClassifier.DEFAULT_WEIGHTS_PATH,
    )
    parser.add_argument("--epochs", type=int, default=3, help="total desired epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
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
