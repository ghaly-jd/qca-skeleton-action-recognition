"""Nearest-neighbor evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.distances.subspace_distances import pairwise_subspace_distances


@dataclass(frozen=True)
class KNNResult:
    """Predictions and nearest-neighbor diagnostics."""

    predictions: np.ndarray
    nearest_indices: np.ndarray
    nearest_distances: np.ndarray


def classify_1nn_subspaces(
    train_bases: np.ndarray,
    train_labels: np.ndarray,
    test_bases: np.ndarray,
    *,
    metric: str,
) -> KNNResult:
    """Classify test subspaces by nearest training subspace."""
    distances = pairwise_subspace_distances(test_bases, train_bases, metric=metric)
    return predict_1nn_from_distances(distances, train_labels)


def predict_1nn_from_distances(
    distance_matrix: np.ndarray,
    train_labels: np.ndarray,
) -> KNNResult:
    """Classify from a precomputed ``N_test x N_train`` distance matrix."""
    distances = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(train_labels)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must have shape N_test x N_train.")
    if distances.shape[1] != labels.shape[0]:
        raise ValueError("train_labels length must match distance_matrix columns.")

    nearest_indices = np.argmin(distances, axis=1)
    return KNNResult(
        predictions=labels[nearest_indices],
        nearest_indices=nearest_indices,
        nearest_distances=distances[np.arange(distances.shape[0]), nearest_indices],
    )
