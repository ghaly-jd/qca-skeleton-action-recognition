from __future__ import annotations

import numpy as np
import pytest

from src.distances.quantum_estimated_angles import (
    exact_subspace_affinity_distance,
    estimate_subspace_squared_overlaps,
    pairwise_exact_subspace_affinity_distances,
    pairwise_quantum_subspace_affinity_distances,
    quantum_subspace_affinity_distance,
    subspace_affinity_from_squared_overlaps,
)
from src.quantum.overlap_estimation import SwapTestOverlapEstimator
from src.quantum.state_preparation import amplitude_encode, amplitude_encode_pair
from src.quantum.swap_test import (
    estimate_squared_overlap_swap_test,
    swap_test_circuit,
    swap_test_zero_probability,
)


def test_amplitude_encoding_pads_and_normalizes_vectors():
    state = amplitude_encode(np.ones(60))

    assert state.original_dim == 60
    assert state.padded_dim == 64
    assert state.num_qubits == 6
    assert np.isclose(np.linalg.norm(state.amplitudes), 1.0)


def test_amplitude_encode_pair_uses_shared_dimension():
    state_x, state_y = amplitude_encode_pair(np.ones(3), np.ones(5))

    assert state_x.padded_dim == state_y.padded_dim == 8
    assert state_x.num_qubits == state_y.num_qubits == 3


def test_swap_test_exact_probability_for_identical_and_orthogonal_vectors():
    basis = np.eye(4)

    assert np.isclose(swap_test_zero_probability(basis[:, 0], basis[:, 0]), 1.0)
    assert np.isclose(swap_test_zero_probability(basis[:, 0], basis[:, 1]), 0.5)

    identical = estimate_squared_overlap_swap_test(
        basis[:, 0],
        basis[:, 0],
        shots=128,
        simulator="exact",
    )
    orthogonal = estimate_squared_overlap_swap_test(
        basis[:, 0],
        basis[:, 1],
        shots=128,
        simulator="exact",
    )

    assert np.isclose(identical.overlap_squared, 1.0)
    assert np.isclose(orthogonal.overlap_squared, 0.0)


def test_swap_test_sampling_is_seeded_and_bounded():
    vector_x = np.asarray([1.0, 1.0])
    vector_y = np.asarray([1.0, -1.0])

    first = estimate_squared_overlap_swap_test(
        vector_x,
        vector_y,
        shots=256,
        simulator="sampling",
        seed=123,
    )
    second = estimate_squared_overlap_swap_test(
        vector_x,
        vector_y,
        shots=256,
        simulator="sampling",
        seed=123,
    )

    assert first.counts == second.counts
    assert 0.0 <= first.overlap_squared <= 1.0


def test_swap_test_circuit_has_expected_register_size():
    pytest.importorskip("qiskit")
    circuit = swap_test_circuit(np.ones(3), np.ones(3))

    assert circuit.num_qubits == 5
    assert circuit.num_clbits == 1


def test_qiskit_aer_swap_test_handles_identical_vectors():
    pytest.importorskip("qiskit_aer")
    result = estimate_squared_overlap_swap_test(
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
        shots=64,
        simulator="qiskit_aer",
        seed=5,
    )

    assert result.counts_zero == 64
    assert np.isclose(result.overlap_squared, 1.0)


def test_quantum_affinity_exact_identical_and_orthogonal_subspaces():
    estimator = SwapTestOverlapEstimator(shots=128, simulator="exact", seed=7)
    basis_x = np.eye(5, 2)
    basis_y = np.eye(5)[:, 2:4]

    assert np.isclose(
        quantum_subspace_affinity_distance(
            basis_x,
            basis_x,
            estimator=estimator,
        ),
        0.0,
    )
    assert np.isclose(
        quantum_subspace_affinity_distance(
            basis_x,
            basis_y,
            estimator=estimator,
        ),
        1.0,
    )


def test_exact_subspace_affinity_matches_quantum_exact_estimator():
    basis_x = np.eye(5, 2)
    basis_y = np.eye(5)[:, 2:4]
    estimator = SwapTestOverlapEstimator(shots=128, simulator="exact", seed=7)

    assert np.isclose(exact_subspace_affinity_distance(basis_x, basis_x), 0.0)
    assert np.isclose(exact_subspace_affinity_distance(basis_x, basis_y), 1.0)
    assert np.isclose(
        exact_subspace_affinity_distance(basis_x, basis_x),
        quantum_subspace_affinity_distance(basis_x, basis_x, estimator=estimator),
    )


def test_subspace_affinity_projection_normalization_sets_identity_to_one():
    overlaps = np.eye(3)

    assert np.isclose(
        subspace_affinity_from_squared_overlaps(
            overlaps,
            normalization="projection_frobenius",
        ),
        1.0,
    )
    assert np.isclose(
        subspace_affinity_from_squared_overlaps(overlaps, normalization="mean"),
        1.0 / 3.0,
    )


def test_pairwise_quantum_affinity_distances_shape_and_values():
    train = np.stack([np.eye(5, 2), np.eye(5)[:, 2:4]], axis=0)
    test = np.stack([np.eye(5, 2)], axis=0)
    estimator = SwapTestOverlapEstimator(shots=128, simulator="exact", seed=11)

    distances = pairwise_quantum_subspace_affinity_distances(
        test,
        train,
        estimator=estimator,
    )
    overlaps = estimate_subspace_squared_overlaps(
        test[0],
        train[0],
        estimator=estimator,
    )

    assert distances.shape == (1, 2)
    assert np.isclose(distances[0, 0], 0.0)
    assert np.isclose(distances[0, 1], 1.0)
    assert np.allclose(overlaps, np.eye(2))


def test_pairwise_exact_affinity_distances_shape_and_values():
    train = np.stack([np.eye(5, 2), np.eye(5)[:, 2:4]], axis=0)
    test = np.stack([np.eye(5, 2)], axis=0)

    distances = pairwise_exact_subspace_affinity_distances(test, train)

    assert distances.shape == (1, 2)
    assert np.isclose(distances[0, 0], 0.0)
    assert np.isclose(distances[0, 1], 1.0)
