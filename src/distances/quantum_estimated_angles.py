"""Quantum-estimated subspace-affinity distances."""

from __future__ import annotations

import numpy as np

from src.quantum.overlap_estimation import SwapTestOverlapEstimator


AFFINITY_NORMALIZATIONS = ["projection_frobenius", "mean"]


def quantum_subspace_affinity_distance(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    *,
    estimator: SwapTestOverlapEstimator,
    normalization: str = "projection_frobenius",
) -> float:
    """Return ``1 - affinity`` from SWAP-test squared-overlap estimates."""
    overlaps = estimate_subspace_squared_overlaps(
        basis_x,
        basis_y,
        estimator=estimator,
    )
    affinity = subspace_affinity_from_squared_overlaps(
        overlaps,
        normalization=normalization,
    )
    return float(max(0.0, 1.0 - affinity))


def estimate_subspace_squared_overlaps(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    *,
    estimator: SwapTestOverlapEstimator,
) -> np.ndarray:
    """Estimate all pairwise basis-vector squared overlaps for two subspaces."""
    x, y = _validate_basis_pair(basis_x, basis_y)
    rank = x.shape[1]
    overlaps = np.empty((rank, rank), dtype=np.float64)
    for i in range(rank):
        for j in range(rank):
            overlaps[i, j] = estimator.estimate(x[:, i], y[:, j]).overlap_squared
    return overlaps


def subspace_affinity_from_squared_overlaps(
    overlaps: np.ndarray,
    *,
    normalization: str = "projection_frobenius",
) -> float:
    """Convert an ``r x r`` squared-overlap matrix into a scalar affinity."""
    values = np.asarray(overlaps, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("overlaps must have shape r x r.")
    rank = values.shape[0]
    if rank == 0:
        raise ValueError("overlaps must be non-empty.")

    normalization = normalization.lower()
    if normalization == "projection_frobenius":
        affinity = float(np.sum(values) / rank)
    elif normalization == "mean":
        affinity = float(np.mean(values))
    else:
        known = ", ".join(AFFINITY_NORMALIZATIONS)
        raise ValueError(f"Unknown affinity normalization '{normalization}': {known}.")
    return float(np.clip(affinity, 0.0, 1.0))


def pairwise_quantum_subspace_affinity_distances(
    test_bases: np.ndarray,
    train_bases: np.ndarray,
    *,
    estimator: SwapTestOverlapEstimator,
    normalization: str = "projection_frobenius",
) -> np.ndarray:
    """Compute a ``N_test x N_train`` quantum-affinity distance matrix."""
    test = _as_basis_stack(test_bases, name="test_bases")
    train = _as_basis_stack(train_bases, name="train_bases")
    if test.shape[1:] != train.shape[1:]:
        raise ValueError("Basis stacks must have matching D x r dimensions.")

    distances = np.empty((test.shape[0], train.shape[0]), dtype=np.float64)
    for test_index in range(test.shape[0]):
        for train_index in range(train.shape[0]):
            distances[test_index, train_index] = quantum_subspace_affinity_distance(
                test[test_index],
                train[train_index],
                estimator=estimator,
                normalization=normalization,
            )
    return distances


def exact_subspace_affinity_distance(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    *,
    normalization: str = "projection_frobenius",
) -> float:
    """Return the exact classical counterpart of quantum affinity distance."""
    x, y = _validate_basis_pair(basis_x, basis_y)
    overlaps = np.square(x.T @ y)
    affinity = subspace_affinity_from_squared_overlaps(
        overlaps,
        normalization=normalization,
    )
    return float(max(0.0, 1.0 - affinity))


def pairwise_exact_subspace_affinity_distances(
    test_bases: np.ndarray,
    train_bases: np.ndarray,
    *,
    normalization: str = "projection_frobenius",
) -> np.ndarray:
    """Compute exact ``1 - subspace_affinity`` distances for basis stacks."""
    test = _as_basis_stack(test_bases, name="test_bases")
    train = _as_basis_stack(train_bases, name="train_bases")
    if test.shape[1:] != train.shape[1:]:
        raise ValueError("Basis stacks must have matching D x r dimensions.")

    cross = np.einsum("tdp,ndq->tnpq", test, train, optimize=True)
    squared_overlaps = np.square(cross)
    normalization = normalization.lower()
    if normalization == "projection_frobenius":
        affinity = np.sum(squared_overlaps, axis=(-2, -1)) / test.shape[2]
    elif normalization == "mean":
        affinity = np.mean(squared_overlaps, axis=(-2, -1))
    else:
        known = ", ".join(AFFINITY_NORMALIZATIONS)
        raise ValueError(f"Unknown affinity normalization '{normalization}': {known}.")
    return np.maximum(0.0, 1.0 - np.clip(affinity, 0.0, 1.0))


def _validate_basis_pair(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(basis_x, dtype=np.float64)
    y = np.asarray(basis_y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Expected D x r basis matrices.")
    if x.shape != y.shape:
        raise ValueError("Basis matrices must have matching D x r shape.")
    if x.shape[1] <= 0:
        raise ValueError("Subspace rank must be positive.")
    return x, y


def _as_basis_stack(values: np.ndarray, *, name: str) -> np.ndarray:
    stack = np.asarray(values, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"{name} must have shape N x D x r.")
    return stack
