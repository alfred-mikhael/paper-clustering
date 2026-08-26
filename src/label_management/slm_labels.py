"""Generate sentence-level weak labels through a llama.cpp server."""

import logging
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import time

import requests
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from ..extract_text import ArxivSection

LABELS = ("A", "B", "C", "D", "E")
MODEL_ID = "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
LABEL_GRAMMAR = 'root ::= "A" | "B" | "C" | "D" | "E"'
CACHE_VERSION = 1

PROMPT_1 = """Classify a paragraph from a mathematical paper by how much it says about a proof technique used by the current paper's authors.

Assign the highest label supported by any statement in the paragraph. Do not average over the paragraph: one strong proof-technique statement is sufficient for a high label.

A: No current-author proof-technique information or extensive description of prior work. This includes results, theorem statements, motivation, definitions, organization, implications, prior work, or other authors' methods.
B: Gives only a high-level description of the authors' approach or lists components of the proof or construction, without an informative technical step.
C: States a concrete operation, reduction, inference, or construction step performed by the authors, but gives little explanation of why it works.
D: Explains how a proof method works, what role it plays, or how it resolves an obstacle.
E: Describes a concrete proof method and explicitly presents that method itself as the main, key, central, crucial, essential, or novel proof idea.

Decision procedure:
1. If a paragraph contains extensive discussion of prior work with no description of the current authors' work or exclusively organizational prose, output A.
2. If a specific proof idea or ingredient is explicitly called main, key, central, crucial, essential, important, or novel to the current authors' proof, output E.
3. Otherwise, if the paragraph explains a method's mechanism, role, purpose, or the obstacle it handles, output D.
4. Otherwise, if it names a concrete proof tool or informative step used by the current authors, output C.
5. Otherwise, if it only refers generically to the current authors' proof, output B.
6. Otherwise, output A.

Rules:
- A description solely of prior authors' methods is A.
- Most paragraphs are A
- A result is not a technique. Statements tagged with theorem, lemma, definition, or corollary are A if not accompanied by more meaningful sentences.
- Judge the strongest proof-technique information in the paragraph.
- Mathematical notation, terminology, and technical vocabulary do not provide meaningful information on their own. 
- If two labels apply, choose the higher one whose conditions are explicitly satisfied.
- Return exactly one letter.

Examples:
Section: Our Work 
Text: In this work, we show a near-cubic lower bound n geq k^3/polylog(k) on the blocklength of any 3 -query LDC. This improves on the previous best lower bound by a O (k) factor. More precisely, we prove: start-of-mtheorem: Let cC colon Bits^k to Bits^n be a code that is (3, delta, eps) -locally decodable. Then, it must hold that k^3 leq n cdot O((log^(6) n)/eps^(32) delta^(16)). In particular, if delta, eps are constants, then n geq Omega(k^3/log^6 k). end-of-mtheorem
Label: A

Section: Proof Overview
Text: This direct argument is not enough. There are 2n possible centers, while the second-moment
estimate gives only a bound of order 2−cn for some small constant c > 0. The latter is not large
enough to support a union bound over all centers. Thus we need a way to upgrade the probability
after paying for only one direct second-moment estimate. We apply the direct second-moment
estimate only once, to a common auxiliary function, and then use that common event to control
all translates with much better conditional probability
Label: D

Section: Our results and ideas
Text: The PCP we present in this paper eliminates the need for compositions in the
sense above.2 As a consequence the honest prover as well as the verifier become significantly simpler and
can be described in a relatively straightforward manner using standard polynomial manipulations. Broadly,
our PCP is obtained by taking the simplest known atomic outer PCP, namely the one from [ABSSW26];
identifying the key bottleneck to a simple composition there, namely the “low-degree test”, and then giving
a new encoding and implementation of the low-degree test that allows for a simple composition. We expand
on these steps by first giving a quick overview of the atomic test.
Label: B

Section: Our results and ideas
Text: The atomic test we use goes back to the work of Babai, Fortnow, and Lund [BFL91], which was first
converted to a “large alphabet” constant query test in [ALMSS98]. These works effectively show that
verifying graph 3-coloring on n vertex graphs “reduces” to verifying that a constant number of m-variate
polynomials over a field Fq are of degree at most d for m, d, q “ polyplog nq and verifying that these
polynomials are zero on some subset of the domain of the form Hm for H Ď Fq. 3 The latter task used to
involve the famed “sum-check protocol” but this aspect (while elegant) is cleaned up significantly in the
work of Ben-Sasson and Sudan [BS08] who show that the final task reduces to m low-degree tests on m
additional polynomials, and further cleaned up by previous work of the authors [ABSSW26] who reduce
this to one low-degree test on just one additional 2m-variate polynomial. The complexity hidden by the
“reductions” alluded to above involves querying a constant number of polynomials at a constant number
of locations and verifying that the query responses are zeroes of a constant number of constant-degree
polynomials.
Label: B

Section: Our Results
Text: Let us begin with our approximate Cauchy-Schwarz inequality. For any functions f, g of
x = (x1, x2, . . . , xn) and any distribution D over x, E_(x sim D) [f(x)g(x)] leq E[f(x)^2]^(1/2)E[g(x)^2]^(1/2). The
proof crucially relies on the positive semidefiniteness of the moment matrix of the pseudo-distribution.
It provably does not hold for Sherali-Adams pseudo-distributions that only satisfy local positive
semidefiniteness. Indeed, in Section 2.1, we observe the following seemingly drastic failure of the Cauchy-Schwarz inequality.
Label: E

Section: Introduction
Text: Up to polylog(k) factors, the best known lower bound of n geq k^( q+1 (q-1))/polylog(k) for q -LDCs for odd q can be obtained by simply observing that a q -LDC is also a (q+1) -LDC, and then invoking the lower bound for (q+1) -query LDCs. Our improvement for q = 3 thus comes from obtaining the same tradeoff with q as in the case of even q, but now for q = 3. For technical reasons, our proof does not extend to odd q geq 5; we briefly mention at the end of the place where the natural generalization fails. We leave proving a lower bound of n geq k^( q (q-2))/polylog(k) for all odd q ge 5 as an intriguing open problem
Label: A

Section: Technical Overview
Text: For this overview, we will assume that the code cC is a linear q -LDC. We will also write the code using Fits notation, so that cC colon Fits^k to Fits^n. By standard reductions (Lemma 6. 2 in [Yek12] ), one can assume that the LDC is in normal form: there exist q -uniform hypergraph matchings cH_1, dots, cH_k, each with Omega(n) hyperedges, and the decoding procedure on input i in [k] simply chooses a uniformly random C in cH_i, and outputs prod_(v in C) x_v. Because cC is linear, when x = cC(b) is the encoding of b, the decoding procedure recovers b_i with probability 1. In other words, for any b in Fits^k, the assignment x = cC(b) satisfies the set of q -XOR constraints forall i in [k], C in cH_i, prod_(v in C) x_v = b_i
Label: A 

Return exactly one letter: A, B, C, D, or E.
Section: {}
Text: {}
Label:"""

PROMPT_2 = """Classify a paragraph from a mathematical paper by how useful it would be for retrieving other papers that use a similar proof technique.

Focus on techniques used by the current paper's authors. The goal is not merely to detect proof-related content, but to identify paragraphs whose mathematical content is specific enough to characterize how the proof works.

A: No useful current-author proof-technique information. This includes theorem statements, results, motivation, definitions, notation, organization, implications, applications, and descriptions of prior work.
B: Indicates that the authors prove something or use an argument, but gives little information that would help identify the technique. Broad descriptions such as "spectral methods," "probabilistic arguments," or "a combinatorial argument" usually belong here.
C: Identifies a concrete proof tool, construction, reduction, or inference that would provide some useful signal for retrieving papers using a similar technique.
D: Gives a substantive and reasonably specific description of a proof mechanism: what is constructed, measured, transformed, bounded, coupled, reduced, iterated, or otherwise done, and possibly why this step is useful. A paragraph at this level should provide strong retrieval information even if some notation depends on surrounding context.
E: Gives an unusually informative and discriminative description of a central proof technique. The paragraph captures enough of the mechanism that papers using a genuinely similar argument should plausibly have semantically similar descriptions. Reserve E for especially valuable technique representations, not merely detailed mathematics.

Important rules:
* Judge retrieval value, not mathematical importance.
* A theorem statement can be very important and still be A.
* A paragraph can deserve D or E without explicitly naming a standard technique if it clearly explains the mechanism.
* Do not reward technical detail that does not help distinguish the proof technique.
* Do not require complete self-containment. A paragraph may use notation defined elsewhere as long as the proof mechanism itself is identifiable.
* Descriptions of methods used only by previous work are always A.
* If it is unclear whether the current authors use the method, choose A.
* When uncertain between two labels, choose the lower one.
* Most paragraphs should receive A or B.

Examples:
Section: Proof Overview
Text: Our proof combines spectral and probabilistic techniques.
Label: B
Reason: It says that techniques are used, but does not identify a specific mechanism useful for retrieval.

Section: Proof Overview
Text: We use expansion properties of the graph to prove the claim.
Label: B
Reason: "Expansion properties" is too broad to identify what kind of expansion argument is being used.

Section: Proof Overview
Text: The main idea is to measure the mixing progress by the l2 norm of the random-walk distribution. Since this norm is minimized by the stationary distribution, we show that it decreases toward its stationary value at every stage.
Label: D
Reason: It explains a specific mechanism for proving mixing and is highly useful for recognizing similar arguments.

Section: Our Results
Text: Let us begin with our approximate Cauchy-Schwarz inequality. For any functions f, g of x = (x1, x2, . . . , xn) and any distribution D over x, E_(x sim D)[f(x)g(x)] leq E[f(x)^2]^(1/2) E[g(x)^2]^(1/2). The proof crucially relies on the positive semidefiniteness of the moment matrix of the pseudo-distribution. It provably does not hold for Sherali-Adams pseudo-distributions that only satisfy local positive semidefiniteness. Indeed, in Section 2.1, we observe the following seemingly drastic failure of the Cauchy-Schwarz inequality.
Label: C
Reason: The paragraph identifies a concrete ingredient of the authors' proof—the positive semidefiniteness of the pseudo-distribution's moment matrix—and distinguishes it from the weaker local PSD property. This gives useful retrieval signal, but it only briefly states the mechanism rather than substantially explaining how the PSD property drives the argument.

Section: Main Results
Text: Theorem 4.3 shows that every graph in the family has expansion at least 0.1.
Label: A
Reason: This is a result, not a description of how it is proved.

Section: This paper: A new lower bound via two reductions
Text: The first reduction relates the maximum load under modular linear hashing (Problem 1) to the maximum load of a continuous version of the problem, real linear hashing (Problem 2). We define the hash functions hreal_a(x) parameterized by a random real number a in [0, 1) as hreal_a(x) = lfloor n (ax) rfloor, where (ax) denotes the fractional part of ax. We show that any lower bound in the real-valued setting that holds for all a in [0, 1) implies the same lower bound in the modular-valued setting that holds for all s, t in Z _p. This reduction allows us to reinterpret known results in combinatorial number theory [konyagin-ruzsa-schlag2000] to immediately obtain a lower bound of exp(Omega(log n / (log log n)^2))
Label: D
Reason: Identifies a reduction between two specific problems and explains what purpose it is used for.

Return only the letter A, B, C, D, or E.
Section: {}
Text: {}
Label: 
"""

USE_PROMPT_1 = False


class SLMWeakLabelGen:
    """Score paragraphs with the model hosted by a llama.cpp server.

    Requests are deliberately sequential so the server can reuse the long common
    prompt prefix from its KV cache. The server must be started with prompt
    caching enabled (the llama.cpp default), preferably with one processing slot.
    """

    def __init__(
        self,
        server_url: str | None = None,
        request_timeout: float = 300.0,
        cache_dir: str | os.PathLike[str] | None = None,
    ):
        load_dotenv()
        self.server_url = (
            server_url or os.getenv("LLAMA_CPP_SERVER_URL") or "http://127.0.0.1:8080"
        ).rstrip("/")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.cache_dir = Path(
            cache_dir or os.getenv("SLM_LABEL_CACHE_DIR") or ".slm_label_cache"
        )

    @staticmethod
    def _probabilities_from_response(response: dict) -> torch.Tensor:
        """Extract and normalize the grammar-filtered A-E probabilities."""
        try:
            token_data = response["choices"][0]["logprobs"]["content"][0]
            candidates = token_data.get("top_probs") or token_data["top_logprobs"]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "llama.cpp response did not contain token probabilities"
            ) from exc

        label_probabilities = {label: 0.0 for label in LABELS}
        for candidate in candidates:
            label = candidate.get("token", "").strip()
            if label not in label_probabilities:
                continue
            if "prob" in candidate:
                probability = float(candidate["prob"])
            elif "logprob" in candidate:
                probability = math.exp(float(candidate["logprob"]))
            else:
                continue
            # More than one token ID can theoretically decode to the same text.
            label_probabilities[label] += probability

        probabilities = torch.tensor(
            [label_probabilities[label] for label in LABELS], dtype=torch.float32
        )
        total = probabilities.sum()
        if not torch.isfinite(probabilities).all() or total <= 0:
            raise RuntimeError(
                "llama.cpp returned no finite probability for labels A-E"
            )
        return probabilities / total

    @staticmethod
    def _structure_message(section, target):
        # This text is repeated for every paragraph, so keeping it compact has a
        # direct effect on prefill time. The rules mirror the original prompt.
        return [
            {
                "role": "user",
                "content": (
                    PROMPT_1.format(section, target)
                    if USE_PROMPT_1
                    else PROMPT_2.format(section, target)
                ),
            }
        ]

    def _cache_details(self, sections: list[ArxivSection]) -> tuple[Path, str]:
        """Return the stable per-paper cache path and its content fingerprint."""
        paper_ids = {section.arxiv_id for section in sections}
        if len(paper_ids) > 1:
            raise ValueError(
                "label_paper expects sections from exactly one arXiv paper"
            )
        paper_id = next(iter(paper_ids), "empty")
        fingerprint_input = json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "model": MODEL_ID,
                "grammar": LABEL_GRAMMAR,
                "prompt": PROMPT_1 if USE_PROMPT_1 else PROMPT_2,
                "sections": [
                    (section.section_index, section.section_header, section.text)
                    for section in sections
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
        filename = paper_id.replace("/", "_")
        return self.cache_dir / f"{filename}.pt", fingerprint

    def _load_cached_labels(
        self, cache_path: Path, fingerprint: str, expected_count: int
    ) -> list[torch.Tensor] | None:
        """Return a valid cached result, or ``None`` if it is stale or corrupt."""
        if not cache_path.exists():
            return None
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            labels = cached["labels"]
            if (
                cached.get("cache_version") != CACHE_VERSION
                or cached.get("fingerprint") != fingerprint
                or not isinstance(labels, torch.Tensor)
                or labels.shape != (expected_count, len(LABELS))
                or not torch.isfinite(labels).all()
            ):
                return None
        except (
            KeyError,
            IndexError,
            TypeError,
            RuntimeError,
            ValueError,
            OSError,
            EOFError,
            pickle.UnpicklingError,
        ):
            logging.warning("Ignoring unreadable SLM label cache at %s", cache_path)
            return None
        logging.info(
            "Loaded %d cached paragraph labels from %s", expected_count, cache_path
        )
        return list(labels.unbind())

    def _save_cached_labels(
        self, cache_path: Path, fingerprint: str, labels: list[torch.Tensor]
    ) -> None:
        """Atomically replace the cache entry after all paragraph requests succeed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        label_tensor = (
            torch.stack(labels).cpu()
            if labels
            else torch.empty((0, len(LABELS)), dtype=torch.float32)
        )
        temporary_path = cache_path.with_suffix(".tmp")
        torch.save(
            {
                "cache_version": CACHE_VERSION,
                "fingerprint": fingerprint,
                "labels": label_tensor,
            },
            temporary_path,
        )
        temporary_path.replace(cache_path)
        logging.info("Cached %d paragraph labels at %s", len(labels), cache_path)

    def label_paper(self, sections: list[ArxivSection]) -> list[torch.Tensor]:
        """Return an A-E probability vector for every paragraph, in paper order."""
        messages = []
        for section in sections:
            for target in section.text:
                messages.append(self._structure_message(section.section_header, target))

        cache_path, fingerprint = self._cache_details(sections)
        cached = self._load_cached_labels(cache_path, fingerprint, len(messages))
        if cached is not None:
            return cached

        results = []
        for message in tqdm(messages, desc="Labeling paragraphs"):
            start = time.perf_counter()
            http_response = None
            try:
                http_response = self.session.post(
                    f"{self.server_url}/v1/chat/completions",
                    json={
                        "messages": message,
                        "max_tokens": 1,
                        "stream": False,
                        "cache_prompt": True,
                        "grammar": LABEL_GRAMMAR,
                        # Neutral sampling settings make the returned distribution
                        # a softmax over the grammar's five allowed labels.
                        "temperature": 1.0,
                        "top_k": 0,
                        "top_p": 1.0,
                        "min_p": 0.0,
                        "typical_p": 1.0,
                        "repeat_penalty": 1.0,
                        "presence_penalty": 0.0,
                        "frequency_penalty": 0.0,
                        "logprobs": True,
                        "top_logprobs": 20,
                        "post_sampling_probs": True,
                    },
                    timeout=self.request_timeout,
                )
                http_response.raise_for_status()
                response = http_response.json()
            except (requests.RequestException, ValueError) as exc:
                detail = ""
                if http_response is not None:
                    detail = f": {http_response.text[:500]}"
                raise RuntimeError(
                    f"Failed to query llama.cpp at {self.server_url}{detail}"
                ) from exc

            results.append(self._probabilities_from_response(response))
            logging.debug(
                "Labeled one paragraph through llama.cpp in %.3fs",
                time.perf_counter() - start,
            )

        self._save_cached_labels(cache_path, fingerprint, results)
        return results


if __name__ == "__main__":
    from ..extract_text import get_sections

    arxiv_id = "2608.24866v1"
    with requests.session() as session:
        paper = get_sections(session, arxiv_id)

    label_gen = SLMWeakLabelGen()
    labels = label_gen.label_paper(paper)

    probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    sentence_scores = []
    for label in labels:
        sentence_scores.append(torch.dot(torch.tensor(probs), label))

    texts = []
    for section in paper:
        texts.extend(section.text)

    for s, t in sorted(zip(sentence_scores, texts)):
        print(s, t)
    # ground_truths = []
    # texts = []
    # import csv

    # with open("./paragraph_labels.tsv", "r") as f:
    #     scorereader = csv.reader(f, delimiter="\t")
    #     next(scorereader)
    #     for row in scorereader:
    #         ground_truths.append(probs[int(row[-1]) - 1])
    #         texts.append(row[-2])

    # print(len(ground_truths), len(sentence_scores))
    # assert len(ground_truths) == len(sentence_scores)
    # n = len(ground_truths)

    # mse = 0
    # for i, (t, y) in enumerate(zip(ground_truths, sentence_scores)):
    #     if (t - y) ** 2 > 0.25:
    #         print(t, y, texts[i])
    #     mse += (t - y) ** 2 / n

    # print(f"MSE = {mse}")
