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


def exact_overlap(u: np.ndarray, v: np.ndarray) -> float:
    return float((u @ v) ** 2)


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


def test_raw_local_sdtw_identical_sequences_have_zero_distance():
    rng = np.random.default_rng(47)
    sequence = rng.normal(size=(18, 6))

    distance = local_sdtw_distance(
        sequence,
        sequence,
        window_size=6,
        stride=3,
        rank=2,
        overlap_fn=exact_overlap,
    )

    assert distance == pytest.approx(0.0, abs=1e-10)


def test_raw_local_sdtw_sequence_vs_reverse_is_positive_and_finite():
    rng = np.random.default_rng(48)
    sequence = rng.normal(size=(21, 6))

    distance = local_sdtw_distance(
        sequence,
        sequence[::-1],
        window_size=5,
        stride=2,
        rank=2,
        overlap_fn=exact_overlap,
    )

    assert np.isfinite(distance)
    assert distance > 0.0


def test_raw_local_sdtw_same_underlying_motion_subspace_is_near_zero():
    rng = np.random.default_rng(49)
    shared_basis, _ = np.linalg.qr(rng.normal(size=(8, 2)))
    first_coefficients = rng.normal(size=(24, 2))
    second_coefficients = rng.normal(size=(24, 2))
    first = first_coefficients @ shared_basis.T
    second = second_coefficients @ shared_basis.T

    distance = local_sdtw_distance(
        first,
        second,
        window_size=6,
        stride=3,
        rank=2,
        overlap_fn=exact_overlap,
    )

    assert distance == pytest.approx(0.0, abs=1e-10)


def test_raw_local_sdtw_orthogonal_motion_subspaces_are_far_apart():
    rng = np.random.default_rng(50)
    first_basis = np.eye(8)[:, :2]
    second_basis = np.eye(8)[:, 2:4]
    first = rng.normal(size=(24, 2)) @ first_basis.T
    second = rng.normal(size=(24, 2)) @ second_basis.T

    distance = local_sdtw_distance(
        first,
        second,
        window_size=6,
        stride=3,
        rank=2,
        overlap_fn=exact_overlap,
    )

    assert distance == pytest.approx(1.0, abs=1e-10)


def test_raw_local_sdtw_is_invariant_to_shared_feature_rotation():
    rng = np.random.default_rng(51)
    first = rng.normal(size=(19, 7))
    second = rng.normal(size=(22, 7))
    rotation, _ = np.linalg.qr(rng.normal(size=(7, 7)))

    original = local_sdtw_distance(
        first,
        second,
        window_size=6,
        stride=3,
        rank=2,
        overlap_fn=exact_overlap,
    )
    rotated = local_sdtw_distance(
        first @ rotation,
        second @ rotation,
        window_size=6,
        stride=3,
        rank=2,
        overlap_fn=exact_overlap,
    )

    assert rotated == pytest.approx(original, abs=1e-10)


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
