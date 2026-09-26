"""Rerank proof passages using OpenJev's NLI cross-encoder.

Both passages form the premise of an explicit technique-overlap hypothesis.
The entailment score measures support for that hypothesis, rather than whether
one paper's passage logically entails the other paper's passage.
"""

from collections.abc import Callable, Sequence
from importlib.util import find_spec
from typing import Literal

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from paper_clustering.utils import Reranker
import logging

logger = logging.getLogger(__name__)


def _quantized_load_options(
    bits: Literal[4, 8] | None, device: torch.device, dtype: torch.dtype
) -> dict[str, object]:
    """Configure optional bitsandbytes loading onto one CUDA device."""
    if bits is None:
        return {}
    if bits not in (4, 8):
        raise ValueError("quantization_bits must be 4, 8, or None")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Quantized reranker loading requires an available CUDA device")
    if find_spec("bitsandbytes") is None or find_spec("accelerate") is None:
        raise ImportError(
            "Quantized reranker loading requires bitsandbytes and accelerate. "
            "Install them with: pip install bitsandbytes accelerate"
        )
    config = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
        # Preserve the classification/language-model heads in floating point.
        llm_int8_skip_modules=["score", "lm_head"],
    )
    return {"quantization_config": config, "device_map": {"": str(device)}}


def _truncate_rerank_passage(tokenizer, text: str, max_length: int = 1900):
    """Keep at most 1900 reranker tokens, including after decoding/re-encoding."""
    token_ids = tokenizer.encode(
        text, padding=False, add_special_tokens=False, return_token_type_ids=False
    )
    if len(token_ids) <= max_length:
        return text
    original_length = len(token_ids)
    token_ids = token_ids[:max_length]
    while True:
        truncated = tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # A decoded token prefix can tokenize differently at its new boundary.
        actual_length = len(
            tokenizer.encode(
                truncated,
                padding=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
        )
        if actual_length <= max_length:
            logger.warning(
                "Truncated reranker passage from %d to %d tokens",
                original_length,
                actual_length,
            )
            return truncated
        token_ids = token_ids[:-1]


class OpenJevReranker(Reranker):
    """Batched technique-overlap scoring with a locally loaded OpenJev model.

    Uses the model's NLI template and label mapping, as documented at
    https://huggingface.co/AlexWortega/openjev. The default checkpoint revision
    is pinned for reproducibility. Loading requires Transformers with Qwen3.5
    sequence-classification support and downloads weights unless cached.

    ``max_length`` limits the entire formatted pair, including the hypothesis.
    Oversized inputs raise ``ValueError`` instead of being silently truncated.
    Scores require validation on reviewed technique pairs before thresholding.

    ``quantization_bits=4`` or ``8`` enables bitsandbytes on a single CUDA
    device and requires bitsandbytes and accelerate. The default ``None``
    preserves unquantized loading. Four-bit loading uses NF4 with double
    quantization; ``dtype`` also controls its computation dtype.
    """

    DEFAULT_MODEL_NAME = "AlexWortega/openjev"
    DEFAULT_SUBFOLDER = "qwen3.5-4b-nli-v2"
    DEFAULT_REVISION = "9f176ba0b528a355a6e574091efcd604ebd8195d"
    NLI_LABELS = ("contradiction", "entailment", "neutral")
    OVERLAP_HYPOTHESIS = (
        "Both passages describe a shared concrete proof mechanism: an "
        "identifiable construction, transformation, or inference used in a "
        "corresponding role. Shared terminology, subject matter, or generic "
        "proof language alone does not establish this."
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        subfolder: str = DEFAULT_SUBFOLDER,
        revision: str | None = DEFAULT_REVISION,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int = 4096,
        quantization_bits: Literal[4, 8] | None = None,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)
        if dtype is None:
            if self.device.type == "cuda":
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            elif self.device.type == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32

        quantized_options = _quantized_load_options(
            quantization_bits, self.device, dtype
        )
        load_options = {"subfolder": subfolder, "revision": revision}
        config = AutoConfig.from_pretrained(model_name, **load_options)
        label_ids = {label.lower(): int(i) for i, label in config.id2label.items()}
        if set(label_ids) != set(self.NLI_LABELS):
            raise ValueError(
                "OpenJev must define contradiction, entailment, and neutral labels"
            )
        self._label_order = [label_ids[label] for label in self.NLI_LABELS]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_options)
        self.tokenizer.padding_side = "right"
        config.pad_token_id = self.tokenizer.pad_token_id
        config.get_text_config().pad_token_id = self.tokenizer.pad_token_id
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, dtype=dtype, **load_options, **quantized_options
        )
        if quantization_bits is None:
            self.model.to(self.device)
        self.model.eval()

    def _format_pair(self, query: str, candidate: str) -> str:
        if not query.strip() or not candidate.strip():
            raise ValueError("Both passages must contain non-whitespace text")
        premise = (
            "Compare the proof mechanisms described in the following passages. "
            "A meaningful overlap requires a concrete shared operation and its "
            "role in the argument, not just a common topic or technique name.\n\n"
            f"Passage A:\n{query.strip()}\n\nPassage B:\n{candidate.strip()}"
        )
        return self.model.config.nli_template.format(
            premise=premise, hypothesis=self.OVERLAP_HYPOTHESIS
        )

    @torch.inference_mode()
    def predict_proba(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 8
    ) -> torch.Tensor:
        """Return an (N, 3) CPU float32 tensor: contradiction, entailment, neutral.

        The classes apply to the overlap hypothesis. In particular, neutrality
        is not a calibrated measure of how much the techniques overlap.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not pairs:
            return torch.empty((0, 3), dtype=torch.float32)

        self.model.eval()
        probabilities = []
        for start in range(0, len(pairs), batch_size):
            texts = [
                self._format_pair(*pair) for pair in pairs[start : start + batch_size]
            ]
            encoded = self.tokenizer(
                texts, padding=False, truncation=False, return_token_type_ids=False
            )
            for offset, token_ids in enumerate(encoded["input_ids"]):
                if len(token_ids) > self.max_length:
                    raise ValueError(
                        f"Passage pair {start + offset} has {len(token_ids)} tokens, "
                        f"exceeding max_length={self.max_length}. Shorten the "
                        "passages or explicitly increase max_length."
                    )
            inputs = self.tokenizer.pad(encoded, padding=True, return_tensors="pt").to(
                self.device
            )
            logits = self.model(**inputs, use_cache=False).logits.float()
            probabilities.append(logits.softmax(dim=-1)[:, self._label_order].cpu())
        return torch.cat(probabilities, dim=0)

    def score_pairs(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 8
    ) -> list[float]:
        """Return entailment scores for the overlap hypothesis in input order."""
        return self.predict_proba(pairs, batch_size=batch_size)[:, 1].tolist()

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        *,
        top_k: int | None = None,
        batch_size: int = 8,
    ) -> list[tuple[int, float]]:
        """Rank candidates, returning original indices and technique-overlap scores."""
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if top_k == 0:
            return []
        scores = self.score_pairs(
            [(query, candidate) for candidate in candidates], batch_size=batch_size
        )
        return sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]


QWEN_RERANK_TEMPLATE = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer "
    'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# Zerank's query/document template, with our instruction prepended to the query.
ZERANK_RERANK_TEMPLATE = (
    "<|im_start|>system\n{instruction}\n\n{query}<|im_end|>\n"
    "<|im_start|>user\n{document}<|im_end|>\n<|im_start|>assistant\n"
)


def qwen_rerank_scores(
    logits: torch.Tensor, tokenizer: PreTrainedTokenizerBase
) -> torch.Tensor:
    """Map (batch, vocabulary) final-token logits to yes/no softmax scores."""
    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("no", "yes")]
    return logits[:, token_ids].float().softmax(dim=-1)[:, 1]


def zerank_rerank_scores(
    logits: torch.Tensor, tokenizer: PreTrainedTokenizerBase
) -> torch.Tensor:
    """Map final-token logits to sigmoid(Yes_logit / 5), per Zerank's model card."""
    token_id = tokenizer.convert_tokens_to_ids("Yes")
    return (logits[:, token_id].float() / 5.0).sigmoid()


class SLMReranker(Reranker):
    """Causal-LM reranking with Qwen3 or Zerank-2 prompt/scoring formats.

    ``template`` is a complete prompt formatted with ``instruction``, ``query``,
    and ``document``. Escape literal braces as {{ and }}. ``scoring_function``
    receives final-token logits of shape (batch, vocabulary) and the tokenizer,
    and must return a tensor of shape (batch,) with finite scores in [0, 1].
    Defaults use Qwen. For Zerank-2, pass ZERANK_RERANK_TEMPLATE and
    zerank_rerank_scores, following
    https://huggingface.co/zeroentropy/zerank-2-reranker.

    Requires Transformers >= 4.51.0. Weights download on construction unless
    cached. Scores are not calibrated probabilities of proof overlap.
    Oversized passages are truncated with logging to fit ``max_length``, while
    preserving the instruction and template. Impossible prompt budgets raise.

    ``quantization_bits=4`` or ``8`` enables bitsandbytes on a single CUDA
    device and requires bitsandbytes and accelerate. The default ``None``
    preserves unquantized loading. Four-bit loading uses NF4 with double
    quantization; ``dtype`` also controls its computation dtype.
    """

    DEFAULT_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
    DEFAULT_INSTRUCTION = (
        "Given a proof passage, retrieve passages sharing a concrete proof "
        "mechanism: an identifiable construction, transformation, or inference "
        "used in a corresponding role. Shared terminology, subject matter, "
        "or generic proof language alone does not establish overlap."
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        revision: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int = 4096,
        quantization_bits: Literal[4, 8] | None = None,
        template: str = QWEN_RERANK_TEMPLATE,
        scoring_function: Callable[
            [torch.Tensor, PreTrainedTokenizerBase], torch.Tensor
        ] = qwen_rerank_scores,
    ) -> None:
        if not template.strip():
            raise ValueError("template must contain non-whitespace text")
        # Validate placeholders before downloading model weights.
        template.format(instruction="", query="", document="")
        if not callable(scoring_function):
            raise TypeError("scoring_function must be callable")
        self.template = template
        self.scoring_function = scoring_function
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if not instruction.strip():
            raise ValueError("instruction must contain non-whitespace text")
        self.instruction = instruction.strip()
        self.max_length = max_length
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)
        if dtype is None:
            if self.device.type == "cuda":
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                dtype = torch.float16 if self.device.type == "mps" else torch.float32

        quantized_options = _quantized_load_options(
            quantization_bits, self.device, dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=dtype, **quantized_options
        )
        if quantization_bits is None:
            self.model.to(self.device)
        if max_length > self.model.config.max_position_embeddings:
            raise ValueError("max_length exceeds the model context length")
        self.model.eval()

    def _format_pair(self, query: str, candidate: str) -> str:
        """Format a pair, truncating passages to fit the full prompt budget."""
        if not query.strip() or not candidate.strip():
            raise ValueError("Both passages must contain non-whitespace text")
        passages = [query.strip(), candidate.strip()]
        while True:
            text = self.template.format(
                instruction=self.instruction,
                query=passages[0],
                document=passages[1],
            )
            prompt_length = len(self.tokenizer.encode(text, add_special_tokens=False))
            if prompt_length <= self.max_length:
                return text

            lengths = [
                len(self.tokenizer.encode(p, add_special_tokens=False))
                for p in passages
            ]
            # Share the available space equally, giving unused space
            # from a short passage to the longer one.
            budget = sum(lengths) - (prompt_length - self.max_length)
            if budget < 2:
                raise ValueError(
                    "max_length is too small to preserve the template, "
                    "instruction, and both passages"
                )
            shorter = 0 if lengths[0] <= lengths[1] else 1
            limits = [0, 0]
            limits[shorter] = min(lengths[shorter], budget // 2)
            limits[1 - shorter] = budget - limits[shorter]
            passages = [
                _truncate_rerank_passage(self.tokenizer, p, max_length=limit)
                for p, limit in zip(passages, limits)
            ]
            if any(not p.strip() for p in passages):
                raise ValueError("Truncation would leave an empty passage")
            # Recheck the complete prompt: token counts can change at
            # the boundaries between passages and template text.

    @torch.inference_mode()
    def score_pairs(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 8
    ) -> list[float]:
        """Return scores in [0, 1] in input order, processing bounded batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not pairs:
            return []
        self.model.eval()
        scores = []
        for start in range(0, len(pairs), batch_size):
            texts = [
                self._format_pair(query, candidate)
                for query, candidate in pairs[start : start + batch_size]
            ]
            encoded = self.tokenizer(
                texts,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            token_batches = []
            for offset, tokens in enumerate(encoded["input_ids"]):
                if len(tokens) > self.max_length:
                    raise ValueError(
                        f"Passage pair {start + offset} has {len(tokens)} tokens, "
                        f"exceeding max_length={self.max_length}. Shorten the "
                        "passages or explicitly increase max_length."
                    )
                token_batches.append(tokens)
            inputs = self.tokenizer.pad(
                {"input_ids": token_batches},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(self.device)
            # Left padding puts the answer position last for every pair.
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits
            batch_scores = self.scoring_function(logits[:, -1, :], self.tokenizer)
            if batch_scores.shape != (len(texts),):
                raise ValueError("scoring_function must return one score per pair")
            if not torch.all(
                torch.isfinite(batch_scores) & (batch_scores >= 0) & (batch_scores <= 1)
            ):
                raise ValueError("scoring_function must return finite scores in [0, 1]")
            scores.extend(batch_scores.cpu().tolist())
        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        *,
        top_k: int | None = None,
        batch_size: int = 8,
    ) -> list[tuple[int, float]]:
        """Return original indices and scores, highest first; ties retain input order."""
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if top_k == 0:
            return []
        scores = self.score_pairs(
            [(query, candidate) for candidate in candidates], batch_size=batch_size
        )
        return sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
