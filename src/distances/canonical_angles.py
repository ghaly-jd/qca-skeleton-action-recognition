"""Canonical-angle computations for subspace bases."""

from __future__ import annotations

import numpy as np


def canonical_angles(basis_x: np.ndarray, basis_y: np.ndarray) -> np.ndarray:
    """Return canonical angles between two orthonormal basis matrices."""
    singular_values = canonical_singular_values(basis_x, basis_y)
    return np.arccos(np.clip(singular_values, -1.0, 1.0))


def canonical_singular_values(basis_x: np.ndarray, basis_y: np.ndarray) -> np.ndarray:
    """Return singular values of ``basis_x.T @ basis_y``."""
    x = _as_basis(basis_x)
    y = _as_basis(basis_y)
    return np.linalg.svd(x.T @ y, compute_uv=False)


def pairwise_canonical_singular_values(
    test_bases: np.ndarray,
    train_bases: np.ndarray,
) -> np.ndarray:
    """Return pairwise canonical singular values as ``N_test x N_train x r``."""
    test = _as_basis_stack(test_bases, name="test_bases")
    train = _as_basis_stack(train_bases, name="train_bases")
    if test.shape[1] != train.shape[1]:
        raise ValueError("Basis stacks must have the same ambient feature dimension.")
    if test.shape[2] != train.shape[2]:
        raise ValueError("Pairwise distances currently require equal subspace rank.")

    cross = np.einsum("tdp,ndq->tnpq", test, train, optimize=True)
    singular_values = np.linalg.svd(cross, compute_uv=False)
    return np.clip(singular_values, -1.0, 1.0)


def _as_basis(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a 2D D x r basis matrix.")
    return values


def _as_basis_stack(values: np.ndarray, *, name: str) -> np.ndarray:
    stack = np.asarray(values, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"{name} must have shape N x D x r.")
    return stack
