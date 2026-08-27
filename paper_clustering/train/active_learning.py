"""Select paragraphs for human labeling with a mix of active-learning strategies."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

LABELS = ("A", "B", "C", "D", "E")
BOUNDARY_WEIGHTS = (0.25, 1.25, 0.75, 0.25)
PREDICTED_FIELDS = tuple(f"label_prob_{label}" for label in LABELS)
GEMMA_FIELD_OPTIONS = (
    tuple(f"gemma_prob_{label}" for label in LABELS),
    tuple(f"gemma_probability_{label}" for label in LABELS),
)
OUTPUT_FIELDS = (
    "strategy",
    "arxiv_id",
    "section_index",
    "paragraph_index",
    "section_header",
    "paragraph",
    *(f"predicted_probability_{label}" for label in LABELS),
    *(f"gemma_probability_{label}" for label in LABELS),
    "predicted_score",
    "entropy",
    "weighted_emd",
)


@dataclass(frozen=True)
class Sample:
    arxiv_id: str
    section_id: int
    paragraph_id: int
    text: str
    predicted_dist: tuple[float, ...]
    gemma_dist: tuple[float, ...]
    section_header: str = ""

    @property
    def key(self) -> tuple[str, int, int]:
        return self.arxiv_id, self.section_id, self.paragraph_id


def _normalize_distribution(
    dist: Sequence[float], *, name: str = "distribution"
) -> tuple[float, ...]:
    values = tuple(float(value) for value in dist)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{name} must contain finite, nonnegative values")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return tuple(value / total for value in values)


def _row_key(row: dict[str, str], line_number: int) -> tuple[str, int, int]:
    try:
        return (
            row["arxiv_id"].strip(),
            int(row["section_index"]),
            int(row["paragraph_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Malformed sample identity on TSV line {line_number}"
        ) from exc


def _load_cached_gemma_distributions(
    cache_dir: Path,
    arxiv_id: str,
    expected_count: int,
) -> list[tuple[float, ...]]:
    """Load document-order A-E distributions from ``SLMWeakLabelGen``'s cache."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to read Gemma label caches; alternatively include "
            "gemma_prob_A through gemma_prob_E in the input TSV"
        ) from exc

    cache_path = cache_dir / f"{arxiv_id.replace('/', '_')}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"No Gemma label cache for {arxiv_id}: {cache_path}")
    try:
        checkpoint = torch.load(cache_path, map_location="cpu", weights_only=True)
        values = checkpoint["labels"].tolist()
    except (KeyError, TypeError, RuntimeError, ValueError, OSError) as exc:
        raise ValueError(f"Unreadable Gemma label cache: {cache_path}") from exc
    if len(values) != expected_count:
        raise ValueError(
            f"Gemma cache for {arxiv_id} contains {len(values)} paragraphs; "
            f"input TSV contains {expected_count}"
        )
    return [
        _normalize_distribution(value, name=f"Gemma distribution for {arxiv_id}")
        for value in values
    ]


def get_samples(
    input_path: Path,
    cache_dir: Path = Path(".slm_label_cache"),
) -> Iterable[Sample]:
    """Read predictions and pair them with Gemma distributions.

    The TSV must contain paragraph identity/text fields and ``label_prob_A``
    through ``label_prob_E``. Gemma probabilities may be included directly as
    ``gemma_prob_A`` ... ``gemma_prob_E``; otherwise the per-paper SLM caches are
    loaded. Cache alignment requires one TSV row for every paragraph in a paper.
    """
    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        required = {
            "arxiv_id",
            "section_index",
            "paragraph_index",
            "paragraph",
            *PREDICTED_FIELDS,
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{input_path} is missing columns: {', '.join(missing)}")
        rows = list(reader)

    parsed_rows = []
    seen_keys = set()
    for line_number, row in enumerate(rows, start=2):
        key = _row_key(row, line_number)
        if not key[0]:
            raise ValueError(f"Empty arXiv ID on TSV line {line_number}")
        if key in seen_keys:
            raise ValueError(
                f"Duplicate paragraph key on TSV line {line_number}: {key}"
            )
        seen_keys.add(key)
        try:
            predicted = _normalize_distribution(
                [row[field] for field in PREDICTED_FIELDS],
                name=f"predicted distribution on TSV line {line_number}",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid predicted probabilities on line {line_number}"
            ) from exc
        parsed_rows.append((key, row, predicted))

    gemma_fields = next(
        (fields for fields in GEMMA_FIELD_OPTIONS if set(fields).issubset(fieldnames)),
        None,
    )
    gemma_by_key: dict[tuple[str, int, int], tuple[float, ...]] = {}
    if gemma_fields is not None:
        for line_number, (key, row, _) in enumerate(parsed_rows, start=2):
            try:
                gemma_by_key[key] = _normalize_distribution(
                    [row[field] for field in gemma_fields],
                    name=f"Gemma distribution on TSV line {line_number}",
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid Gemma probabilities on line {line_number}"
                ) from exc
    else:
        rows_by_paper: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for key, _, _ in parsed_rows:
            rows_by_paper[key[0]].append(key)
        for arxiv_id, keys in rows_by_paper.items():
            keys.sort(key=lambda key: (key[1], key[2]))
            cached = _load_cached_gemma_distributions(cache_dir, arxiv_id, len(keys))
            gemma_by_key.update(zip(keys, cached))

    for key, row, predicted in parsed_rows:
        yield Sample(
            arxiv_id=key[0],
            section_id=key[1],
            paragraph_id=key[2],
            section_header=row.get("section_header", "").strip(),
            text=row["paragraph"].strip(),
            predicted_dist=predicted,
            gemma_dist=gemma_by_key[key],
        )


def calculate_entropy(dist: tuple[float, ...]) -> float:
    """Return Shannon entropy using natural logarithms; zero-mass terms are safe."""
    normalized = _normalize_distribution(dist)
    return -sum(
        probability * math.log(probability) for probability in normalized if probability
    )


def expected_score(dist: tuple[float, ...]) -> float:
    """Return the expected ordinal A-E score, where A=1 and E=5."""
    normalized = _normalize_distribution(dist)
    return sum(index * probability for index, probability in enumerate(normalized, 1))


def _validate_selection_count(n: int) -> None:
    if n < 0:
        raise ValueError("selection count must be nonnegative")


def get_highest_entropy(samples: list[Sample], n: int) -> list[Sample]:
    """Return up to ``n`` samples with the greatest predictive entropy."""
    _validate_selection_count(n)
    return sorted(
        samples,
        key=lambda sample: (-calculate_entropy(sample.predicted_dist), sample.key),
    )[:n]


def get_paper_topk(samples: list[Sample], arxiv_id: str, k: int) -> list[Sample]:
    """Return a paper's top ``k`` paragraphs by expected predicted A-E score."""
    _validate_selection_count(k)
    paper_samples = [sample for sample in samples if sample.arxiv_id == arxiv_id]
    return sorted(
        paper_samples,
        key=lambda sample: (-expected_score(sample.predicted_dist), sample.key),
    )[:k]


def get_random_from_topk(
    samples: list[Sample],
    k: int,
    n: int,
    *,
    rng: random.Random | None = None,
) -> list[Sample]:
    """Sample from the union of every paper's top-``k`` paragraphs."""
    _validate_selection_count(k)
    _validate_selection_count(n)
    generator = rng or random.Random()
    paper_ids = sorted({sample.arxiv_id for sample in samples})
    candidates = [
        sample
        for arxiv_id in paper_ids
        for sample in get_paper_topk(samples, arxiv_id, k)
    ]
    return generator.sample(candidates, k=min(n, len(candidates)))


def get_random(
    samples: list[Sample], n: int, *, rng: random.Random | None = None
) -> list[Sample]:
    """Return up to ``n`` uniformly sampled paragraphs."""
    _validate_selection_count(n)
    generator = rng or random.Random()
    return generator.sample(samples, k=min(n, len(samples)))


def calculate_weighted_emd(dist1: tuple[float, ...], dist2: tuple[float, ...]) -> float:
    """Return weighted one-dimensional EMD across the ordered A-E boundaries."""
    first = _normalize_distribution(dist1, name="first distribution")
    second = _normalize_distribution(dist2, name="second distribution")
    if len(first) != len(second):
        raise ValueError("distributions must have the same length")
    if len(BOUNDARY_WEIGHTS) != len(first) - 1:
        raise ValueError("boundary weights do not match the distributions")

    cumulative_difference = 0.0
    distance = 0.0
    for boundary, weight in enumerate(BOUNDARY_WEIGHTS):
        cumulative_difference += first[boundary] - second[boundary]
        distance += weight * abs(cumulative_difference)
    return distance


def get_max_disagreement(samples: list[Sample], n: int) -> list[Sample]:
    """Return samples with the greatest weighted EMD from Gemma's distribution."""
    _validate_selection_count(n)
    return sorted(
        samples,
        key=lambda sample: (
            -calculate_weighted_emd(sample.predicted_dist, sample.gemma_dist),
            sample.key,
        ),
    )[:n]


def select_samples(
    samples: list[Sample],
    *,
    entropy_count: int = 0,
    disagreement_count: int = 0,
    topk_random_count: int = 0,
    topk_per_paper: int = 5,
    random_count: int = 0,
    seed: int = 42,
) -> list[tuple[str, Sample]]:
    """Combine strategies in order while preventing duplicate selections."""
    for count in (
        entropy_count,
        disagreement_count,
        topk_random_count,
        topk_per_paper,
        random_count,
    ):
        _validate_selection_count(count)

    selected: list[tuple[str, Sample]] = []
    selected_keys = set()
    rng = random.Random(seed)

    def available() -> list[Sample]:
        return [sample for sample in samples if sample.key not in selected_keys]

    def add(strategy: str, chosen: Iterable[Sample]) -> None:
        for sample in chosen:
            if sample.key not in selected_keys:
                selected.append((strategy, sample))
                selected_keys.add(sample.key)

    add("entropy", get_highest_entropy(available(), entropy_count))
    add("disagreement", get_max_disagreement(available(), disagreement_count))
    add(
        "topk_random",
        get_random_from_topk(available(), topk_per_paper, topk_random_count, rng=rng),
    )
    add("random", get_random(available(), random_count, rng=rng))
    return selected


def save_selected(path: Path, selected: list[tuple[str, Sample]]) -> None:
    """Atomically save selected paragraphs and diagnostic selection scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for strategy, sample in selected:
            writer.writerow(
                {
                    "strategy": strategy,
                    "arxiv_id": sample.arxiv_id,
                    "section_index": sample.section_id,
                    "paragraph_index": sample.paragraph_id,
                    "section_header": sample.section_header,
                    "paragraph": sample.text,
                    **{
                        f"predicted_probability_{label}": f"{value:.9g}"
                        for label, value in zip(LABELS, sample.predicted_dist)
                    },
                    **{
                        f"gemma_probability_{label}": f"{value:.9g}"
                        for label, value in zip(LABELS, sample.gemma_dist)
                    },
                    "predicted_score": f"{expected_score(sample.predicted_dist):.9g}",
                    "entropy": f"{calculate_entropy(sample.predicted_dist):.9g}",
                    "weighted_emd": (
                        f"{calculate_weighted_emd(sample.predicted_dist, sample.gemma_dist):.9g}"
                    ),
                }
            )
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select model-predicted paragraphs for human labeling"
    )
    parser.add_argument("input", type=Path, help="TSV containing model predictions")
    parser.add_argument("output", type=Path, help="selected paragraph TSV")
    parser.add_argument("--slm-cache-dir", type=Path, default=Path(".slm_label_cache"))
    parser.add_argument("--entropy-count", type=int, default=0)
    parser.add_argument("--disagreement-count", type=int, default=0)
    parser.add_argument("--topk-random-count", type=int, default=0)
    parser.add_argument("--topk-per-paper", type=int, default=5)
    parser.add_argument("--random-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    counts = (
        args.entropy_count,
        args.disagreement_count,
        args.topk_random_count,
        args.random_count,
    )
    if all(count == 0 for count in counts):
        parser.error("request at least one sample with a --*-count option")
    try:
        samples = list(get_samples(args.input, args.slm_cache_dir))
        selected = select_samples(
            samples,
            entropy_count=args.entropy_count,
            disagreement_count=args.disagreement_count,
            topk_random_count=args.topk_random_count,
            topk_per_paper=args.topk_per_paper,
            random_count=args.random_count,
            seed=args.seed,
        )
        save_selected(args.output, selected)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Selected {len(selected)} of {len(samples)} paragraphs into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
