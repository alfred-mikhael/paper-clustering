"""Cluster paper embedding vectors."""

import numpy as np


def _as_matrix(data: list[np.ndarray]) -> np.ndarray:
    """Validate a collection of equal-length, finite embedding vectors."""
    if not data:
        return np.empty((0, 0), dtype=float)

    try:
        matrix = np.asarray(data, dtype=float)
    except ValueError as error:
        raise ValueError(
            "data must contain one-dimensional vectors of the same length"
        ) from error
    if matrix.ndim != 2:
        raise ValueError("data must contain one-dimensional vectors of the same length")
    if not np.isfinite(matrix).all():
        raise ValueError("data vectors must contain only finite values")
    return matrix


def generate_clusters(
    data: list[np.ndarray], strategy: str = "hdbscan"
) -> list[int]:
    """Assign a density-based cluster label to every vector in ``data``.

    The ``"hdbscan"`` strategy uses scikit-learn's HDBSCAN implementation.
    Cluster labels are non-negative integers and noise points have label ``-1``.
    """
    matrix = _as_matrix(data)
    if len(matrix) == 0:
        return []

    if strategy.casefold() != "hdbscan":
        raise ValueError(f"Unsupported clustering strategy: {strategy!r}")

    # HDBSCAN's default minimum cluster size is five, so fewer inputs cannot
    # form a cluster under the default configuration.
    if len(matrix) < 5:
        return [-1] * len(matrix)

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as error:
        raise ImportError(
            "The 'hdbscan' strategy requires scikit-learn 1.3 or later."
        ) from error

    return [int(label) for label in HDBSCAN().fit_predict(matrix)]
