"""Subspace distances derived from canonical angles."""

from __future__ import annotations

import numpy as np

from src.distances.canonical_angles import (
    canonical_angles,
    pairwise_canonical_singular_values,
)


DISTANCE_NAMES = ["chordal", "projection", "mean_angle", "max_angle", "min_angle"]


def subspace_distance(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    *,
    metric: str = "chordal",
) -> float:
    """Compute one named distance between two subspaces."""
    angles = canonical_angles(basis_x, basis_y)
    singular_values = np.cos(angles)
    return float(_distance_from_singular_values(singular_values, metric=metric))


def pairwise_subspace_distances(
    test_bases: np.ndarray,
    train_bases: np.ndarray,
    *,
    metric: str = "chordal",
) -> np.ndarray:
    """Compute a ``N_test x N_train`` distance matrix."""
    singular_values = pairwise_canonical_singular_values(test_bases, train_bases)
    return distance_from_singular_values(singular_values, metric=metric)


def distance_from_singular_values(
    singular_values: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """Compute a named subspace distance from canonical singular values."""
    return _distance_from_singular_values(singular_values, metric=metric)


def _distance_from_singular_values(
    singular_values: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    metric = metric.lower()
    sigma = np.clip(np.asarray(singular_values, dtype=np.float64), -1.0, 1.0)
    rank = sigma.shape[-1]
    sin_squared = np.maximum(0.0, 1.0 - sigma**2)

    if metric == "chordal":
        return np.sqrt(np.sum(sin_squared, axis=-1))
    if metric == "projection":
        return np.sqrt(np.maximum(0.0, 2.0 * rank - 2.0 * np.sum(sigma**2, axis=-1)))
    if metric == "mean_angle":
        return np.mean(np.arccos(sigma), axis=-1)
    if metric == "max_angle":
        return np.max(np.arccos(sigma), axis=-1)
    if metric == "min_angle":
        return np.min(np.arccos(sigma), axis=-1)

    known = ", ".join(DISTANCE_NAMES)
    raise ValueError(f"Unknown subspace distance '{metric}'. Known values: {known}.")
