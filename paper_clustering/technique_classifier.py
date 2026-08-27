"""Neural classifier for detecting proof-technique information in paragraphs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


class TechniqueClassifier(ABC):
    """Interface for models that score proof-technique passages."""

    @abstractmethod
    def predict(self, texts: list[str]) -> torch.Tensor:
        """Return a score from 0 to 4 for each supplied passage."""


class SciBERTClassifier(nn.Module, TechniqueClassifier):
    """SciBERT followed by a small MLP for ordinal paragraph classification.

    SciBERT converts each token in a paragraph into a contextual representation.
    We use the representation of the special ``[CLS]`` token as a fixed-size
    summary of the complete paragraph, then pass that summary through an MLP.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier for the encoder. The default is the
        uncased SciBERT checkpoint with its scientific-domain vocabulary.
    mlp_hidden_size:
        Width of the MLP's hidden layer. This is intentionally much smaller
        than SciBERT's hidden size so the head learns a compact task-specific
        representation without adding many parameters.
    dropout:
        Probability of dropping an activation in the classification head during
        training. Dropout is automatically disabled when ``model.eval()`` is
        called.
    weights_path:
        Optional checkpoint to load after constructing the model. The file may
        contain either a plain PyTorch state dictionary or the richer checkpoint
        written by ``train_classifier.py``. Pass ``None`` to start from the
        pretrained SciBERT checkpoint even if the default local checkpoint exists.
    """

    DEFAULT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
    DEFAULT_WEIGHTS_PATH = Path("weights/technique_classifier_checkpoint.pt")

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        mlp_hidden_size: int = 256,
        dropout: float = 0.1,
        weights_path: str | Path | None = DEFAULT_WEIGHTS_PATH,
    ) -> None:
        super().__init__()

        if mlp_hidden_size < 1:
            raise ValueError("mlp_hidden_size must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.scibert = AutoModel.from_pretrained(model_name)
        encoder_hidden_size = self.scibert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 5),
        )
        self.classifier.apply(self._initialize_head_weights)

        self.weights_path = Path(weights_path) if weights_path is not None else None
        if self.weights_path is not None and self.weights_path.exists():
            self.load_weights(self.weights_path)

    def _initialize_head_weights(self, module: nn.Module) -> None:
        """Initialize MLP linear layers consistently with the BERT checkpoint."""
        if isinstance(module, nn.Linear):
            initializer_range = getattr(self.scibert.config, "initializer_range", 0.02)
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def load_weights(self, weights_path: str | Path) -> None:
        """Load model parameters from a plain or training-harness checkpoint."""
        checkpoint = torch.load(
            Path(weights_path), map_location="cpu", weights_only=True
        )

        # The training harness stores additional information alongside the
        # model parameters. A directly saved model.state_dict(), on the other
        # hand, is already the mapping expected by load_state_dict(). Supporting
        # both formats keeps this class useful outside the supplied harness.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid model checkpoint: {weights_path}")
        self.load_state_dict(state_dict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_output = self.scibert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_representation = encoder_output.last_hidden_state[:, 0, :]
        return self.classifier(cls_representation).squeeze(-1)

    def predict(self, passages: list[str]) -> torch.Tensor:
        """Predict an ordinal score from 0 to 4 for every passage."""
        if not passages:
            return torch.empty(0, dtype=torch.long)

        device = next(self.parameters()).device
        batches = DataLoader(passages, batch_size=16, shuffle=False)
        predictions: list[torch.Tensor] = []
        was_training = self.training

        self.eval()
        try:
            with torch.inference_mode():
                for batch in batches:
                    inputs = self.tokenizer(
                        list(batch),
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    model_inputs = {
                        key: value.to(device)
                        for key, value in inputs.items()
                        if key in {"input_ids", "attention_mask", "token_type_ids"}
                    }
                    logits = self(**model_inputs)
                    predictions.append(logits.argmax(dim=-1).cpu())
        finally:
            self.train(was_training)

        return torch.cat(predictions)
