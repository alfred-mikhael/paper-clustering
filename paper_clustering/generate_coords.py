"""Generate 2d coordinates for visualizing a set of high-dimensional vectors."""

from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Protocol

import numpy as np
from paper_clustering.data_models import EmbeddedPaper


class UMAPReducer(Protocol):
    """A fitted UMAP reducer capable of projecting new vectors."""

    def transform(self, data: np.ndarray) -> np.ndarray: ...


def _as_matrix(data: list[np.ndarray] | list[EmbeddedPaper]) -> np.ndarray:
    """Extract area vectors from papers and validate a homogeneous vector matrix."""
    if not data:
        return np.empty((0, 0), dtype=float)

    vectors = [
        item.area_vector if isinstance(item, EmbeddedPaper) else item for item in data
    ]
    try:
        matrix = np.asarray(vectors, dtype=float)
    except ValueError as error:
        raise ValueError(
            "data must contain one-dimensional vectors of the same length"
        ) from error
    if matrix.ndim != 2:
        raise ValueError("data must contain one-dimensional vectors of the same length")
    if not np.isfinite(matrix).all():
        raise ValueError("data vectors must contain only finite values")
    return matrix


def _coordinates(matrix: np.ndarray) -> list[tuple[float, float]]:
    """Convert a two-column coordinate array into the public return type."""
    return [tuple(map(float, point)) for point in matrix]


def generate_umap(
    data: Iterable[np.ndarray | EmbeddedPaper],
    weights_path: str | Path | None = None,
    *,
    save_weights_path: str | Path | None = None,
    reducer: UMAPReducer | None = None,
    batch_size: int | None = None,
) -> Iterable[tuple[float, float]]:
    """Yield coordinates in input order, loading and batching in one place.

    ``weights_path`` must point to a UMAP reducer serialized with ``joblib``.
    Alternatively, pass an already-loaded ``reducer`` to reuse it across batches.
    With neither supplied, fit a new reducer to ``data`` and optionally save it
    to ``save_weights_path``. Saving is only supported when fitting. Fitting
    consumes the full input; transformation consumes batches (default: 1000).
    An explicit ``batch_size`` is only valid with a pretrained reducer.
    Validation and computation run when the returned iterable is consumed.
    """
    if weights_path is not None and reducer is not None:
        raise ValueError("Supply either weights_path or reducer, not both")
    fitting = weights_path is None and reducer is None
    if save_weights_path is not None and not fitting:
        raise ValueError("save_weights_path is only valid when fitting")
    if batch_size is not None:
        if fitting:
            raise ValueError("batch_size requires a pretrained reducer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

    if weights_path is not None:
        import joblib

        reducer = joblib.load(Path(weights_path))
    items = iter(data)
    vector_dim = None
    while True:
        batch = list(items) if fitting else list(islice(items, batch_size or 1000))
        if not batch:
            if fitting and save_weights_path is not None:
                raise ValueError("Fitting UMAP requires at least four papers")
            return
        matrix = _as_matrix(batch)
        if matrix.shape[1] == 0:
            raise ValueError("Area vectors must be nonempty")
        if vector_dim is not None and matrix.shape[1] != vector_dim:
            raise ValueError("Area vector dimensions differ between batches")
        vector_dim = matrix.shape[1]
        if fitting:
            if len(matrix) < 4:
                raise ValueError("Fitting UMAP requires at least four papers")
            from umap import UMAP

            fitted_reducer = UMAP(
                n_components=2,
                metric="cosine",
                random_state=42,
            )
            reduced = fitted_reducer.fit_transform(matrix)
        else:
            reduced = reducer.transform(matrix)

        reduced = np.asarray(reduced)
        if reduced.shape != (len(matrix), 2) or not np.isfinite(reduced).all():
            raise ValueError("UMAP must return two finite coordinates per paper")
        if save_weights_path is not None:
            import joblib

            destination = Path(save_weights_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted_reducer, destination)

        yield from _coordinates(reduced)
        if fitting:
            return


def generate_tsne(
    data: list[np.ndarray] | list[EmbeddedPaper],
) -> list[tuple[float, float]]:
    """Generate deterministic t-SNE coordinates from vectors or paper area embeddings."""
    matrix = _as_matrix(data)

    from sklearn.manifold import TSNE

    # t-SNE requires perplexity to be smaller than the number of samples.
    perplexity = min(30, max(1, (len(matrix) - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        random_state=42,
    )
    return _coordinates(reducer.fit_transform(matrix))
