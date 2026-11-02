"""Centroid arithmetic, cluster validation, and condensed-tree interpretation."""

import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class ClusterAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stubs = {}
        if importlib.util.find_spec("hdbscan") is None:
            module = ModuleType("hdbscan")
            module.HDBSCAN = Mock(side_effect=AssertionError("unexpected model fit"))
            stubs["hdbscan"] = module
        with patch.dict(sys.modules, stubs):
            cls.cluster = importlib.import_module("paper_clustering.cluster")

    def paper(self, paper_id, vector=(1.0, 0.0), model="test"):
        return SimpleNamespace(
            metadata=SimpleNamespace(arxiv_id=paper_id, title=paper_id, abstract="Abstract"),
            area_vector=np.array(vector), embedding_dim=len(vector), model_name=model,
        )

    def test_centroids_deduplicate_memberships_and_keep_cluster_order(self):
        memberships = [
            {"cluster_id": "2", "arxiv_id": "a"}, {"cluster_id": "2", "arxiv_id": "a"},
            {"cluster_id": "2", "arxiv_id": "b"}, {"cluster_id": "1", "arxiv_id": "b"},
        ]
        coordinates = [{"arxiv_id": "a", "umap_x": "0", "umap_y": "2"},
                       {"arxiv_id": "b", "umap_x": "4", "umap_y": "6"}]
        self.assertEqual(self.cluster.approximate_centroid(memberships, coordinates),
                         [(2, (2.0, 4.0)), (1, (4.0, 6.0))])
        self.assertEqual(
            self.cluster.approximate_centroid(memberships, coordinates, num_samples=1, random_state=42),
            self.cluster.approximate_centroid(memberships, coordinates, num_samples=1, random_state=42),
        )

    def test_centroids_reject_missing_duplicate_and_nonfinite_coordinates(self):
        point = {"arxiv_id": "a", "umap_x": 0, "umap_y": 0}
        for memberships, coordinates in (
            ([{"cluster_id": 1, "arxiv_id": "missing"}], [point]),
            ([], [point, point]), ([], [{**point, "umap_x": float("nan")}]),
            ([], [{**point, "umap_y": "not numeric"}]),
        ):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                self.cluster.approximate_centroid(memberships, coordinates)
        for count in (0, True, 1.5):
            with self.assertRaises(ValueError):
                self.cluster.approximate_centroid([], [], num_samples=count)

    def test_invalid_vectors_and_incompatible_models_fail_before_fitting(self):
        for papers in (
            [self.paper("a"), self.paper("a")], [self.paper("a", ())],
            [self.paper("a", (float("nan"), 0))],
            [self.paper("a"), self.paper("b", model="other")],
            [self.paper("a"), self.paper("b", (1, 2, 3))],
        ):
            with self.subTest(papers=papers), self.assertRaises(ValueError):
                self.cluster.validate_cluster_papers(papers)
        with patch.object(self.cluster, "HDBSCAN") as fit:
            self.assertEqual(self.cluster.generate_clusters([self.paper("a")]), ([], {}))
        fit.assert_not_called()

    def test_condensed_tree_expands_ancestor_memberships_and_keeps_probabilities(self):
        tree = np.array([(3, 0, 1), (3, 4, 2), (4, 1, 1), (4, 2, 1)],
                        dtype=[("parent", int), ("child", int), ("child_size", int)])
        model = Mock()
        model.fit.return_value = model
        model.condensed_tree_.to_numpy.return_value = tree
        model.probabilities_ = np.array([0.1, 0.8, 0.9])
        with patch.object(self.cluster, "HDBSCAN", return_value=model):
            records, clusters = self.cluster.generate_clusters(
                [self.paper("a"), self.paper("b"), self.paper("c")], min_cluster_size=2,
            )
        self.assertEqual([(row.cluster_id, row.arxiv_id) for row in records],
                         [(3, "a"), (3, "b"), (3, "c"), (4, "b"), (4, "c")])
        self.assertEqual([row.certainty for row in records], [0.1, 0.8, 0.9, 0.8, 0.9])
        self.assertEqual(clusters[3].size, 3)
        self.assertEqual(clusters[3].parent_cluster_id, -1)
        self.assertEqual(clusters[4].parent_cluster_id, 3)

    def test_invalid_clustering_and_labelling_options_fail(self):
        for settings in ({"min_cluster_size": 1}, {"min_samples": 0}, {"umap_dim": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.cluster.validate_clustering_options(**settings)
        for settings in ({"num_samples": 0}, {"request_timeout": float("inf")}, {"request_timeout": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.cluster.validate_labelling_options(**settings)


if __name__ == "__main__":
    unittest.main()
