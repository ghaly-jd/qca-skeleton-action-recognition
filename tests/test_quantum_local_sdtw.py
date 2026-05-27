from __future__ import annotations

import numpy as np
import pytest

from src.distances.local_sdtw import local_sdtw_distance, local_subspace_cost
from src.distances.quantum_local_sdtw import (
    pairwise_quantum_exact_local_sdtw_comparison,
    pairwise_quantum_local_sdtw_distances,
    quantum_local_sdtw_distance,
    quantum_local_subspace_cost,
    quantum_local_subspace_cost_matrix,
)
from src.features.local_subspace import compute_local_subspace_sequence
from src.quantum.overlap_estimation import SwapTestOverlapEstimator


def test_quantum_local_cost_exact_matches_projection_affinity_cost():
    basis_x = np.eye(4, 2)
    basis_y = np.eye(4)[:, 1:3]
    estimator = SwapTestOverlapEstimator(shots=128, simulator="exact", seed=7)

    assert quantum_local_subspace_cost(
        basis_x,
        basis_y,
        estimator=estimator,
    ) == pytest.approx(local_subspace_cost(basis_x, basis_y))


def test_identical_local_sequences_have_zero_quantum_distance_in_exact_mode():
    rng = np.random.default_rng(10)
    sequence = compute_local_subspace_sequence(
        rng.normal(size=(14, 5)),
        sequence_id="seq",
        rank=2,
        window_length=6,
        stride=3,
    )
    estimator = SwapTestOverlapEstimator(shots=128, simulator="exact", seed=8)

    assert quantum_local_sdtw_distance(
        sequence,
        sequence,
        estimator=estimator,
    ) == pytest.approx(0.0, abs=1e-10)


def test_sampling_quantum_local_cost_matrix_is_bounded():
    rng = np.random.default_rng(11)
    first = compute_local_subspace_sequence(
        rng.normal(size=(12, 5)),
        sequence_id="first",
        rank=1,
        window_length=5,
        stride=4,
    )
    second = compute_local_subspace_sequence(
        rng.normal(size=(13, 5)),
        sequence_id="second",
        rank=1,
        window_length=5,
        stride=4,
    )
    estimator = SwapTestOverlapEstimator(shots=64, simulator="sampling", seed=9)

    costs = quantum_local_subspace_cost_matrix(first, second, estimator=estimator)

    assert costs.shape == (3, 3)
    assert np.all(costs >= 0.0)
    assert np.all(costs <= 1.0)


def test_more_shots_reduces_error_on_fixed_toy_pair():
    rng = np.random.default_rng(12)
    first = compute_local_subspace_sequence(
        rng.normal(size=(12, 5)),
        sequence_id="first",
        rank=1,
        window_length=5,
        stride=4,
    )
    second = compute_local_subspace_sequence(
        rng.normal(size=(12, 5)),
        sequence_id="second",
        rank=1,
        window_length=5,
        stride=4,
    )
    exact = local_sdtw_distance(first, second)
    low_shot = quantum_local_sdtw_distance(
        first,
        second,
        estimator=SwapTestOverlapEstimator(shots=64, simulator="sampling", seed=10),
    )
    high_shot = quantum_local_sdtw_distance(
        first,
        second,
        estimator=SwapTestOverlapEstimator(shots=4096, simulator="sampling", seed=10),
    )

    assert abs(high_shot - exact) <= abs(low_shot - exact) + 1e-12


def test_pairwise_quantum_local_sdtw_distance_matrix_shape():
    rng = np.random.default_rng(13)
    sequences = [
        compute_local_subspace_sequence(
            rng.normal(size=(12 + index, 5)),
            sequence_id=f"seq{index}",
            rank=1,
            window_length=5,
            stride=4,
        )
        for index in range(3)
    ]
    estimator = SwapTestOverlapEstimator(shots=128, simulator="exact", seed=11)

    distances = pairwise_quantum_local_sdtw_distances(
        sequences[:1],
        sequences[1:],
        estimator=estimator,
    )

    assert distances.shape == (1, 2)
    assert np.all(distances >= 0.0)


def test_pairwise_quantum_exact_comparison_reports_diagnostics():
    rng = np.random.default_rng(14)
    sequences = [
        compute_local_subspace_sequence(
            rng.normal(size=(12 + index, 5)),
            sequence_id=f"seq{index}",
            rank=1,
            window_length=5,
            stride=4,
        )
        for index in range(3)
    ]
    estimator = SwapTestOverlapEstimator(shots=128, simulator="exact", seed=12)

    comparison = pairwise_quantum_exact_local_sdtw_comparison(
        sequences[:1],
        sequences[1:],
        estimator=estimator,
    )

    assert comparison.exact_distances.shape == (1, 2)
    np.testing.assert_allclose(
        comparison.quantum_distances,
        comparison.exact_distances,
        atol=1e-10,
    )
    assert len(comparison.pair_diagnostics) == 2
    assert comparison.num_dtw_pairs == 2
    assert comparison.num_local_costs > 0
