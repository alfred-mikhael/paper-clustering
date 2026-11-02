"""Coordinate batching through real UMAP orchestration and CSV storage."""

from contextlib import contextmanager
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

from paper_clustering import pipeline
from paper_clustering.local_storage import LocalStorage


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.storage = LocalStorage()
        self.reducer = Mock()
        self.reducer.transform.side_effect = lambda matrix: matrix[:, :2]
        self.joblib = ModuleType("joblib")
        self.joblib.load = Mock(return_value=self.reducer)
        self.joblib.dump = Mock()
        for patcher in (
            patch.dict(sys.modules, {"joblib": self.joblib}),
            patch.object(pipeline, "local_storage", self.storage),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_local_transform_is_bounded_and_keeps_metadata_aligned(self):
        def rows():
            for index in range(5):
                self.assertEqual(self.reducer.transform.call_count, index // 2)
                yield {"arxiv_id": str(index), "title": f"Title {index}"}, np.array([index, -index])

        output = self.root / "coords.csv.gz"
        with patch.object(self.storage, "iter_embedding_rows", side_effect=lambda *args: rows()), patch.object(
            pipeline, "_database_client"
        ) as connect:
            pipeline.generate_coordinates(
                output, input_filepath=self.root, metadata_filepath=self.root,
                umap_weights=self.root / "weights", batch_size=2,
            )
        connect.assert_not_called()
        self.joblib.load.assert_called_once()
        self.assertEqual([len(call.args[0]) for call in self.reducer.transform.call_args_list], [2, 2, 1])
        records = self.storage.load_coordinates(output)
        self.assertEqual([row["arxiv_id"] for row in records], [str(i) for i in range(5)])
        self.assertEqual([float(row["umap_y"]) for row in records], [0, -1, -2, -3, -4])

    def test_database_read_finishes_before_transform(self):
        released = []
        client = Mock()
        client.select.return_value = [
            {"arxiv_id": "a", "title": "A", "area_embedding": [1.0, 2.0]}
        ]

        @contextmanager
        def connect():
            yield client
            released.append(True)

        def transform(matrix):
            self.assertEqual(released, [True])
            return matrix

        self.reducer.transform.side_effect = transform
        with patch.object(pipeline, "_database_client", side_effect=connect):
            pipeline.generate_coordinates(self.root / "coords.gz", umap_weights=self.root / "weights")
        client.select.assert_called_once()

    def test_vector_dimensions_are_checked_across_batches(self):
        rows = [("a", "A", [1, 2]), ("b", "B", [1, 2, 3])]
        with patch.object(pipeline, "_coordinate_rows", return_value=iter(rows)):
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                pipeline.generate_coordinates(
                    self.root / "coords.gz", umap_weights=self.root / "weights", batch_size=1,
                )

    def test_fitting_still_consumes_all_vectors_together(self):
        umap = ModuleType("umap")
        reducer = Mock()
        reducer.fit_transform.side_effect = lambda matrix: matrix[:, :2]
        umap.UMAP = Mock(return_value=reducer)
        rows = [(str(i), str(i), [i, i + 1]) for i in range(4)]
        with patch.dict(sys.modules, {"umap": umap}), patch.object(
            pipeline, "_coordinate_rows", return_value=iter(rows)
        ):
            pipeline.generate_coordinates(
                self.root / "coords.gz", save_umap_weights=self.root / "weights",
            )
        reducer.fit_transform.assert_called_once()
        self.assertEqual(len(reducer.fit_transform.call_args.args[0]), 4)
        self.joblib.dump.assert_called_once()


if __name__ == "__main__":
    unittest.main()
