import numpy as np
import pytest

from src.distances.local_sdtw import (
    dtw_distance_from_cost_matrix,
    local_sdtw_distance,
    local_subspace_cost,
    local_subspace_cost_matrix,
    pairwise_local_sdtw_distances,
)
from src.features.local_subspace import compute_local_subspace_sequence


def test_projection_affinity_cost_for_identical_and_orthogonal_bases():
    basis_x = np.eye(3, 1)
    basis_y = np.eye(3, 1)
    basis_z = np.asarray([[0.0], [1.0], [0.0]])

    assert local_subspace_cost(basis_x, basis_y) == pytest.approx(0.0)
    assert local_subspace_cost(basis_x, basis_z) == pytest.approx(1.0)


def test_local_subspace_cost_rejects_unknown_metric():
    basis = np.eye(3, 1)

    with pytest.raises(ValueError, match="Unknown local subspace distance"):
        local_subspace_cost(basis, basis, metric="mystery")


def test_dtw_distance_from_cost_matrix_normalizes_by_path_length():
    costs = np.asarray(
        [
            [1.0, 5.0],
            [5.0, 1.0],
        ]
    )

    unnormalized = dtw_distance_from_cost_matrix(
        costs,
        normalize_by_path_length=False,
    )
    normalized = dtw_distance_from_cost_matrix(
        costs,
        normalize_by_path_length=True,
    )

    assert unnormalized == pytest.approx(2.0)
    assert normalized == pytest.approx(1.0)


def test_dtw_cost_matrix_window_none_matches_none_string():
    costs = np.asarray(
        [
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )

    assert dtw_distance_from_cost_matrix(costs, window_ratio=None) == pytest.approx(
        dtw_distance_from_cost_matrix(costs, window_ratio="none")
    )


def test_identical_local_subspace_sequences_have_zero_distance():
    rng = np.random.default_rng(42)
    sequence = rng.normal(size=(16, 5))
    local_sequence = compute_local_subspace_sequence(
        sequence,
        sequence_id="x",
        rank=2,
        window_length=6,
        stride=3,
    )

    distance = local_sdtw_distance(local_sequence, local_sequence)

    assert distance == pytest.approx(0.0, abs=1e-10)


def test_local_subspace_cost_matrix_has_expected_shape():
    rng = np.random.default_rng(43)
    first = compute_local_subspace_sequence(
        rng.normal(size=(16, 5)),
        sequence_id="first",
        rank=2,
        window_length=6,
        stride=3,
    )
    second = compute_local_subspace_sequence(
        rng.normal(size=(12, 5)),
        sequence_id="second",
        rank=2,
        window_length=6,
        stride=3,
    )

    costs = local_subspace_cost_matrix(first, second)

    assert costs.shape == (4, 3)
    assert np.all(costs >= 0.0)


def test_pairwise_local_sdtw_distances_shape_and_nearest_identity():
    rng = np.random.default_rng(44)
    raw_sequences = [rng.normal(size=(14, 5)) for _ in range(3)]
    local_sequences = [
        compute_local_subspace_sequence(
            sequence,
            sequence_id=f"seq{index}",
            rank=2,
            window_length=6,
            stride=4,
        )
        for index, sequence in enumerate(raw_sequences)
    ]

    distances = pairwise_local_sdtw_distances(local_sequences, local_sequences)

    assert distances.shape == (3, 3)
    np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-10)
    assert np.array_equal(np.argmin(distances, axis=1), np.arange(3))


def test_torch_cpu_local_sdtw_matches_numpy_projection_affinity():
    pytest.importorskip("torch")
    rng = np.random.default_rng(45)
    raw_sequences = [rng.normal(size=(15 + index, 5)) for index in range(4)]
    local_sequences = [
        compute_local_subspace_sequence(
            sequence,
            sequence_id=f"seq{index}",
            rank=2,
            window_length=6,
            stride=3,
        )
        for index, sequence in enumerate(raw_sequences)
    ]

    expected = pairwise_local_sdtw_distances(
        local_sequences[:2],
        local_sequences[2:],
        backend="numpy",
        window_ratio=0.2,
    )
    actual = pairwise_local_sdtw_distances(
        local_sequences[:2],
        local_sequences[2:],
        backend="torch_cpu",
        torch_dtype="float64",
        window_ratio=0.2,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_torch_local_sdtw_rejects_non_projection_affinity_metric():
    pytest.importorskip("torch")
    rng = np.random.default_rng(46)
    local_sequence = compute_local_subspace_sequence(
        rng.normal(size=(12, 5)),
        sequence_id="seq",
        rank=2,
        window_length=6,
        stride=3,
    )

    with pytest.raises(ValueError, match="supports only metric='projection_affinity'"):
        pairwise_local_sdtw_distances(
            [local_sequence],
            [local_sequence],
            metric="chordal",
            backend="torch_cpu",
        )
