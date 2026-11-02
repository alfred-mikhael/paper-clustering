"""Sampling quality, reproducibility, and paragraph identity preservation."""

import math
import random
from types import SimpleNamespace
import unittest

from paper_clustering.data_models import ArxivSection, Paper
from paper_clustering.train.active_learning import (
    Sample, calculate_entropy, calculate_weighted_emd, expected_score,
    get_highest_entropy, get_max_disagreement, get_paper_topk, get_random_from_topk,
    get_regex_weighted_random, select_samples, select_samples_from_paper,
)


class ActiveLearningTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            Sample("a", 0, 0, "First", (1, 0, 0, 0, 0), (1, 0, 0, 0, 0)),
            Sample("a", 0, 1, "Second", (1, 1, 1, 1, 1), (1, 0, 0, 0, 0)),
            Sample("b", 1, 0, "Third", (0, 0, 0, 0, 1), (1, 0, 0, 0, 0)),
        ]

    def test_entropy_expected_score_and_weighted_distance_have_known_values(self):
        self.assertAlmostEqual(calculate_entropy((1, 1, 1, 1, 1)), math.log(5))
        self.assertEqual(calculate_entropy((0, 0, 1, 0, 0)), 0)
        self.assertEqual(expected_score((0, 0, 0, 0, 2)), 5)
        self.assertEqual(calculate_weighted_emd((1, 0, 0, 0, 0), (0, 0, 0, 0, 1)), 2.5)
        self.assertEqual(calculate_weighted_emd((0, 1, 0, 0, 0), (0, 0, 1, 0, 0)), 1.25)
        for values in ((), (0, 0), (-1, 2), (float("nan"), 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                calculate_entropy(values)
        for other in ((1, 0), (1, 0, 0, 0)):
            with self.assertRaises(ValueError):
                calculate_weighted_emd((1, 0, 0, 0, 0), other)

    def test_strategies_rank_expected_samples(self):
        self.assertEqual(get_highest_entropy(self.samples, 1), [self.samples[1]])
        self.assertEqual(get_max_disagreement(self.samples, 1), [self.samples[2]])
        self.assertEqual(get_paper_topk(self.samples, "a", 1), [self.samples[1]])
        self.assertEqual(get_paper_topk(self.samples, "missing", 1), [])

    def test_topk_random_only_selects_each_papers_top_candidates(self):
        selected = get_random_from_topk(self.samples, 1, 20, rng=random.Random(42))
        self.assertEqual({sample.key for sample in selected}, {self.samples[1].key, self.samples[2].key})

    def test_mixed_strategies_are_repeatable_without_duplicate_selections(self):
        arguments = dict(entropy_count=1, disagreement_count=1, random_count=10, seed=5)
        selected = select_samples(self.samples, **arguments)
        self.assertEqual(selected, select_samples(self.samples, **arguments))
        self.assertEqual([strategy for strategy, _ in selected], ["entropy", "disagreement", "random"])
        self.assertEqual(len({sample.key for _, sample in selected}), 3)
        with self.assertRaises(ValueError):
            select_samples(self.samples, entropy_count=-1)

    def test_regex_weighted_sampling_falls_back_without_replacement(self):
        selected = get_regex_weighted_random(self.samples, 10, rng=random.Random(42))
        self.assertEqual({sample.key for sample in selected}, {sample.key for sample in self.samples})
        self.assertEqual(len(selected), 3)

    def test_paper_selection_preserves_sections_and_normalizes_probabilities(self):
        paper = Paper(SimpleNamespace(arxiv_id="a"), (
            ArxivSection("a", 3, "Methods", "Methods", ("First", "Second")),
        ))
        probabilities = [[0, 0, 0, 0, 2], [1, 1, 1, 1, 1]]
        selected = select_samples_from_paper(paper, probabilities, probabilities, random_count=2)
        self.assertEqual({sample.key for _, sample in selected}, {("a", 3, 0), ("a", 3, 1)})
        self.assertTrue(all(sample.section_header == "Methods" for _, sample in selected))
        self.assertTrue(all(sum(sample.predicted_dist) == 1 for _, sample in selected))
        for predicted, gemma in (([], probabilities), (probabilities, []), ([[1], [1]], probabilities)):
            with self.assertRaises(ValueError):
                select_samples_from_paper(paper, predicted, gemma)


if __name__ == "__main__":
    unittest.main()
