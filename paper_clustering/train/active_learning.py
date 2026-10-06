"""Select paragraphs with a mix of active-learning strategies."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..data_models import ArxivSection, Paper
from ..label.regex_labels import extract_paragraph_features

BOUNDARY_WEIGHTS = (0.25, 1.25, 0.75, 0.25)


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
    """Validate and normalize a probability distribution for selection."""
    values = tuple(float(value) for value in dist)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{name} must contain finite, nonnegative values")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return tuple(value / total for value in values)


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


def get_regex_features(samples: Sequence[Sample]) -> list[tuple[int, int, int]]:
    """Generate important-section, enrichment, and negative regex features."""
    sections = [
        ArxivSection(
            arxiv_id=sample.arxiv_id,
            section_index=sample.section_id,
            section_header=sample.section_header,
            major_section_header="",
            text=(sample.text,),
        )
        for sample in samples
    ]
    return [tuple(features) for features in extract_paragraph_features(sections)]


def get_regex_weighted_random(
    samples: list[Sample], n: int, *, rng: random.Random | None = None
) -> list[Sample]:
    """Sample without replacement, weighted by regex evidence and section type.

    Each paragraph's weight is ``enrichment_pattern_hits + 3 *
    important_section_title``. If no paragraph has positive weight, the
    remaining selections fall back to uniform sampling.
    """
    _validate_selection_count(n)
    generator = rng or random.Random()
    candidates = list(zip(samples, get_regex_features(samples)))
    selected = []
    while candidates and len(selected) < n:
        weights = [
            enrichment_hits + 3 * important_section
            for _, (important_section, enrichment_hits, _) in candidates
        ]
        total_weight = sum(weights)
        if total_weight == 0:
            index = generator.randrange(len(candidates))
        else:
            threshold = generator.random() * total_weight
            cumulative_weight = 0.0
            index = len(candidates) - 1
            for candidate_index, weight in enumerate(weights):
                cumulative_weight += weight
                if threshold < cumulative_weight:
                    index = candidate_index
                    break
        sample, _ = candidates.pop(index)
        selected.append(sample)
    return selected


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
    regex_weighted_random_count: int = 0,
    seed: int = 42,
) -> list[tuple[str, Sample]]:
    """Combine strategies in order while preventing duplicate selections."""
    for count in (
        entropy_count,
        disagreement_count,
        topk_random_count,
        topk_per_paper,
        random_count,
        regex_weighted_random_count,
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
    add(
        "regex_weighted_random",
        get_regex_weighted_random(
            available(), regex_weighted_random_count, rng=rng
        ),
    )
    add("random", get_random(available(), random_count, rng=rng))
    return selected


def select_samples_from_paper(
    paper: Paper,
    predicted_distributions: Sequence[Sequence[float]],
    gemma_distributions: Sequence[Sequence[float]],
    *,
    entropy_count: int = 0,
    disagreement_count: int = 0,
    topk_random_count: int = 0,
    topk_per_paper: int = 5,
    random_count: int = 0,
    regex_weighted_random_count: int = 0,
    seed: int = 42,
) -> list[tuple[str, Sample]]:
    """Select a paper's paragraphs from document-order model distributions.

    ``predicted_distributions`` and ``gemma_distributions`` must each provide
    one A-E distribution for every paragraph in ``paper.sections`` order.
    """
    paragraphs = [
        (section, paragraph_index, text)
        for section in paper.sections
        for paragraph_index, text in enumerate(section.text)
    ]
    if len(predicted_distributions) != len(paragraphs):
        raise ValueError(
            "predicted_distributions must contain one distribution per paragraph"
        )
    if len(gemma_distributions) != len(paragraphs):
        raise ValueError(
            "gemma_distributions must contain one distribution per paragraph"
        )

    samples = []
    for index, ((section, paragraph_index, text), predicted, gemma) in enumerate(
        zip(paragraphs, predicted_distributions, gemma_distributions), start=1
    ):
        if section.arxiv_id != paper.metadata.arxiv_id:
            raise ValueError(
                "every paper section must have the same arXiv ID as its metadata"
            )
        predicted_dist = _normalize_distribution(
            predicted, name=f"predicted distribution for paragraph {index}"
        )
        gemma_dist = _normalize_distribution(
            gemma, name=f"Gemma distribution for paragraph {index}"
        )
        if len(predicted_dist) != 5 or len(gemma_dist) != 5:
            raise ValueError(
                "each predicted and Gemma distribution must have five values"
            )
        samples.append(
            Sample(
                arxiv_id=section.arxiv_id,
                section_id=section.section_index,
                paragraph_id=paragraph_index,
                section_header=section.section_header,
                text=text,
                predicted_dist=predicted_dist,
                gemma_dist=gemma_dist,
            )
        )

    return select_samples(
        samples,
        entropy_count=entropy_count,
        disagreement_count=disagreement_count,
        topk_random_count=topk_random_count,
        topk_per_paper=topk_per_paper,
        random_count=random_count,
        regex_weighted_random_count=regex_weighted_random_count,
        seed=seed,
    )
