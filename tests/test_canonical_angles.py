from __future__ import annotations

import numpy as np

from src.distances.canonical_angles import canonical_angles
from src.distances.subspace_distances import (
    pairwise_subspace_distances,
    subspace_distance,
)


def test_identical_subspaces_have_zero_chordal_distance():
    basis = np.eye(5, 2)

    assert np.isclose(subspace_distance(basis, basis, metric="chordal"), 0.0)
    assert np.allclose(canonical_angles(basis, basis), [0.0, 0.0])


def test_orthogonal_subspaces_have_large_distance():
    basis_x = np.eye(6, 2)
    basis_y = np.eye(6)[:, 2:4]

    assert np.allclose(canonical_angles(basis_x, basis_y), [np.pi / 2, np.pi / 2])
    assert np.isclose(subspace_distance(basis_x, basis_y, metric="chordal"), np.sqrt(2.0))
    assert np.isclose(subspace_distance(basis_x, basis_y, metric="projection"), 2.0)


def test_basis_sign_flips_do_not_change_distances():
    basis_x = np.eye(5, 2)
    basis_y = basis_x.copy()
    basis_y[:, 0] *= -1.0

    assert np.isclose(subspace_distance(basis_x, basis_y, metric="mean_angle"), 0.0)
    assert np.isclose(subspace_distance(basis_x, basis_y, metric="max_angle"), 0.0)


def test_pairwise_subspace_distances_shape_and_values():
    train = np.stack([np.eye(5, 2), np.eye(5)[:, 2:4]], axis=0)
    test = np.stack([np.eye(5, 2)], axis=0)

    distances = pairwise_subspace_distances(test, train, metric="chordal")

    assert distances.shape == (1, 2)
    assert np.isclose(distances[0, 0], 0.0)
    assert distances[0, 1] > 1.0
