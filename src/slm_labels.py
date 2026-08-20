"""Generate sentence-level weak labels with a small causal language model."""

import logging
import os
import time
from collections.abc import Iterator, Sequence

import spacy
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

from .extract_intro import ArxivSection

LABELS = ("A", "B", "C", "D", "E")

PROMPT_1 = """Classify only <target> from a mathematical paper by how much it says about a proof technique used by the current paper's authors. Use context only for references and attribution; do not credit information found only in context.

A: No current-author proof-technique information (including results, theorem statements, motivation, definitions, organization, implications, prior work, or other authors' methods).
B: Says the authors prove something or use an argument, but names no technique or informative proof step.
C: Identifies a specific local proof tool, construction, reduction, or inference.
D: Explains a substantive proof mechanism and possibly its role or the obstacle it handles.
E: Identifies a specific strategy, construction, intermediate result, or mathematical structure as explicitly central, essential, crucial, key, or novel in the current authors' proof.

Rules: Prior work is always A. A result is not a technique. Notation or technical vocabulary alone does not raise the label. If authorship is unclear choose A; if torn choose the lower label. E requires explicit evidence of centrality. Most labels should be A or B. Description of how a theorem is proved should be labelled at least C.
Apply the labels in this order:
1. Choose E if a specific proof ingredient is explicitly called central,
   essential, crucial, key, main, or novel.
2. Otherwise choose D if the target explains a substantive mechanism or role.
3. Otherwise choose C if it identifies a specific local tool, construction,
   reduction, or inference.
4. Otherwise choose B if it refers generically to the authors' proof.
5. Otherwise choose A.

Examples:
<context-before>Our approach builds on the framework introduced by Chen and Rao. They associate an auxiliary graph with each admissible configuration.</context-before>
<target>Their proof then applies spectral partitioning to find a sparse cut and proceeds by induction on the resulting components.</target>
<context-after>We instead analyze the auxiliary graph without decomposing it.</context-after>
Label: A

<context-before>We can now state our main quantitative result.</context-before>
<target>We prove that every linear four-query locally decodable code of message length k has block length at least exp(Ω(k^{{1/3}})).</target>
<context-after>This improves the exponent in the previous lower bound.</context-after>
Label: A

<context-before>A direct concentration argument is dominated by the few vertices of very high degree.</context-before>
<target>We partition the vertices into dyadic degree classes and apply concentration separately within each class, allowing the error to be charged to the class size rather than to the maximum degree.</target>
<context-after>Summing the resulting estimates over the degree classes proves the lemma.</context-after>
Label: D

<context-before>For any functions f,g of x=(x_1,x_2,...,x_n) and any distribution D over x, E_(x sim D)[f(x)g(x)] leq E[f(x)^2]^{{1/2}}E[g(x)^2]^{{1/2}}</context-before>
<target>The proof crucially relies on the positive semidefiniteness of the moment matrix of the pseudo-distribution</target>
<context-after>It provably does not hold for Sherali-Adams pseudo-distributions that only satisfy local positive semidefiniteness</context-after>
Label: E

Return exactly one letter: A, B, C, D, or E.
Section: {}
<context-before>{}</context-before>
<target>{}</target>
<context-after>{}</context-after>
Label:"""
PROMPT_2 = """Classify only <target> from a mathematical paper according to how much it reveals about a proof technique used by the current paper's authors. Use context only to resolve references and attribution. Do not credit information stated only in context.

A: Little or no current-author proof-technique information. This includes results, theorem statements, motivation, nonessential definitions, notation, organization, implications, applications, and descriptions of prior work.

B: Weak proof-technique information. The sentence refers to the authors' proof, argument, construction, or tools, but gives little detail. This includes generic proof descriptions, isolated proof steps, and merely naming or listing tools without explaining an important role.

C: Strong proof-technique information. The sentence identifies a specific technique, tool, construction, or sequence of steps as central to the proof; explains a substantive mechanism or why it works; explains how an obstacle is overcome; or describes the technical idea responsible for an improvement over prior work.

Rules:
- Classify only information stated in <target>.
- Use context only to determine attribution and resolve expressions such as “their proof,” “this argument,” or “the construction.”
- A description solely of prior authors' methods is always A.
- A bare claim that the authors prove, establish, improve, or obtain a result is A.
- Merely naming mathematical objects or using technical vocabulary is A unless they are described as part of the current authors' proof.
- Merely naming or listing a proof tool is usually B. Assign C only when the target establishes centrality, explains its role or mechanism, gives meaningful detail, or identifies it as the source of an improvement.
- Saying that a result improves prior work is A; explaining the proof idea that enables the improvement may be C.
- If authorship is unclear, choose A. If torn between labels, choose the lower label.
- Most sentences should receive A.

Now classify:

Section: {}
<context-before>{}</context-before>
<target>{}</target>
<context-after>{}</context-after>
Label:"""
USE_PROMPT_1 = True


class SLMWeakLabelGen:
    """Score sentences without autoregressively generating an answer.

    The model only needs to predict the first response token because every valid
    response is a single letter. This is substantially faster than ``generate``.
    """

    def __init__(self, load_4bit: bool = True):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")

        self.model_id = "google/gemma-3-4b-it"
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # bitsandbytes 4-bit kernels require CUDA.
        use_4bit = load_4bit and self.device.type == "cuda"
        if self.device.type == "cuda":
            # Gemma's output projection can overflow in float16 and produce
            # all-NaN label probabilities. Prefer bfloat16 even when the model
            # weights themselves are stored in 4-bit.
            model_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16 if use_4bit else torch.float32
            )
        elif self.device.type == "mps":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.bfloat16
        self.bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=model_dtype,
            )
            if use_4bit
            else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
        # The final position must be a real prompt token for every item. With
        # right padding, ``logits[:, -1]`` would score padding for short inputs.
        self.tokenizer.padding_side = "left"

        model_kwargs = {
            "token": hf_token,
            "quantization_config": self.bnb_config,
            "dtype": model_dtype,
            "device_map": str(self.device),
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **model_kwargs
        ).eval()
        self.parser = spacy.load("en_core_web_sm", disable=["ner", "tagger"])

        label_ids = [
            self.tokenizer.encode(label, add_special_tokens=False) for label in LABELS
        ]
        if any(len(ids) != 1 for ids in label_ids):
            raise ValueError("Each label must map to exactly one tokenizer token")
        self.label_ids = torch.tensor([ids[0] for ids in label_ids], device=self.device)

    @staticmethod
    def _structure_message(section, prev, target, next_):
        # This text is repeated for every sentence, so keeping it compact has a
        # direct effect on prefill time. The rules mirror the original prompt.
        return [
            {
                "role": "user",
                "content": (
                    PROMPT_1.format(section, prev, target, next_)
                    if USE_PROMPT_1
                    else PROMPT_2.format(section, prev, target, next_)
                ),
            }
        ]

    def _format_message(self, section, prev, target, next_) -> str:
        return self.tokenizer.apply_chat_template(
            self._structure_message(section, prev, target, next_),
            tokenize=False,
            add_generation_prompt=True,
        )

    def _format_message_batch(self, messages: Sequence[str]):
        return self.tokenizer(
            messages,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8 if self.device.type == "cuda" else None,
            add_special_tokens=True,
        ).to(self.device, non_blocking=self.device.type == "cuda")

    @staticmethod
    def _batches_by_length(
        messages: Sequence[str], batch_size: int
    ) -> Iterator[list[tuple[int, str]]]:
        """Group similarly sized prompts to avoid doing work on padding.

        Indices are retained so results can be restored to document order.
        Character count is a cheap and sufficiently accurate token-count proxy.
        """
        indexed = sorted(enumerate(messages), key=lambda item: len(item[1]))
        for start in range(0, len(indexed), batch_size):
            yield indexed[start : start + batch_size]

    def label_paper(
        self, sections: list[ArxivSection], batch_size: int = 8
    ) -> list[torch.Tensor]:
        """Return an A-E probability vector for every sentence, in paper order."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        messages = []
        for section in sections:
            sentences = [sent.text for sent in self.parser(section.text).sents]
            for index, target in enumerate(sentences):
                prev = sentences[index - 1] if index else ""
                next_ = sentences[index + 1] if index + 1 < len(sentences) else ""
                messages.append(
                    self._format_message(section.section_header, prev, target, next_)
                )

        if not messages:
            return []

        # Store probabilities on CPU. Keeping each result as a CUDA view slowly
        # consumes accelerator memory when labeling many papers.
        results: list[torch.Tensor | None] = [None] * len(messages)
        batches = self._batches_by_length(messages, batch_size)
        total_batches = (len(messages) + batch_size - 1) // batch_size

        with torch.inference_mode():
            for indexed_batch in tqdm(batches, total=total_batches):
                indices, batch_messages = zip(*indexed_batch)
                batch = self._format_message_batch(batch_messages)

                start = time.perf_counter()
                # There is no decoding step, so building a KV cache wastes both
                # time and memory. logits_to_keep avoids materializing logits for
                # every prompt position on supported Transformers versions.
                output = self.model(
                    **batch,
                    use_cache=False,
                    logits_to_keep=1,
                )
                # Transfer only five logits per sentence. Checking and applying
                # softmax on CPU avoids introducing an extra CUDA synchronization.
                next_logits = output.logits[:, -1, :].float().cpu()
                label_logits = output.logits[:, -1, self.label_ids].float().cpu()
                if not torch.isfinite(label_logits).all():
                    raise FloatingPointError(
                        "Gemma produced non-finite label logits. On CUDA, use a "
                        "GPU with bfloat16 support or load the model without 4-bit "
                        "quantization; float16 can overflow in Gemma's LM head."
                    )
                probabilities = torch.softmax(label_logits, dim=-1)
                top_log_probs, top_ids = torch.softmax(next_logits, dim=-1).topk(
                    3, dim=-1
                )

                for row_ids, row_log_probs in zip(top_ids, top_log_probs):
                    print(
                        [
                            (
                                repr(self.tokenizer.decode([token_id.item()])),
                                log_probability.exp().item(),
                            )
                            for token_id, log_probability in zip(row_ids, row_log_probs)
                        ]
                    )
                    print(
                        [
                            re.findall(r"<target>(.*)</target>", msg)[-1]
                            for msg in batch_messages
                        ]
                    )
                elapsed = time.perf_counter() - start
                logging.debug(
                    "Labeled %d sentences (%d padded tokens) in %.3fs",
                    len(indexed_batch),
                    batch["input_ids"].numel(),
                    elapsed,
                )

                for original_index, probabilities_for_sentence in zip(
                    indices, probabilities
                ):
                    results[original_index] = probabilities_for_sentence

        return [result for result in results if result is not None]


if __name__ == "__main__":
    import requests
    import pprint
    from .extract_intro import get_sections

    with requests.session() as session:
        paper = get_sections(session, "2404.05864")

    label_gen = SLMWeakLabelGen()
    labels = label_gen.label_paper([paper[0]], batch_size=8)
    # pprint.pp(paper)
    # print(labels)
