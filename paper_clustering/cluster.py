"""Cluster paper embedding vectors."""

import numpy as np


def generate_clusters(data: list[np.ndarray], strategy: str = "hdbscan") -> list[int]:
    """Assign a density-based cluster label to every vector in ``data``.

    The ``"hdbscan"`` strategy uses the HDBScan implementation in ``hdsbscan``.
    Cluster labels are non-negative integers and noise points have label ``-1``.
    """
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
