import numpy as np
import pytest

from src.features.local_subspace import (
    compute_local_subspace_sequence,
    make_motion_features,
    make_sliding_windows,
)
from src.features.sequence_subspace import is_orthonormal


def test_make_sliding_windows_includes_tail_when_needed():
    sequence = np.arange(13 * 2, dtype=float).reshape(13, 2)

    windows = make_sliding_windows(sequence, window_length=5, stride=4)

    assert [(start, end) for start, end, _ in windows] == [(0, 5), (4, 9), (8, 13)]


def test_short_sequence_uses_single_window_by_default():
    sequence = np.arange(4 * 3, dtype=float).reshape(4, 3)

    windows = make_sliding_windows(sequence, window_length=10, stride=5)

    assert len(windows) == 1
    assert windows[0][0] == 0
    assert windows[0][1] == 4
    np.testing.assert_allclose(windows[0][2], sequence)


def test_short_sequence_error_mode_raises():
    sequence = np.arange(4 * 3, dtype=float).reshape(4, 3)

    with pytest.raises(ValueError, match="below window_length"):
        make_sliding_windows(
            sequence,
            window_length=10,
            stride=5,
            short_sequence_mode="error",
        )


def test_compute_local_subspace_sequence_returns_orthonormal_bases():
    rng = np.random.default_rng(123)
    sequence = rng.normal(size=(20, 6))

    local_sequence = compute_local_subspace_sequence(
        sequence,
        sequence_id="toy",
        rank=2,
        window_length=8,
        stride=4,
    )

    assert local_sequence.sequence_id == "toy"
    assert local_sequence.window_length == 8
    assert local_sequence.stride == 4
    assert local_sequence.rank == 2
    assert len(local_sequence.windows) == 4
    for window in local_sequence.windows:
        assert window.basis.shape == (6, 2)
        assert window.singular_values.shape == (2,)
        assert window.num_frames == 8
        assert is_orthonormal(window.basis)


def test_compute_local_subspace_sequence_rejects_invalid_rank_for_window():
    sequence = np.arange(4 * 6, dtype=float).reshape(4, 6)

    with pytest.raises(ValueError, match="need more than rank=4"):
        compute_local_subspace_sequence(
            sequence,
            sequence_id="short",
            rank=4,
            window_length=4,
            stride=2,
        )


def test_velocity_and_position_velocity_features_have_expected_shapes():
    sequence = np.arange(5 * 3, dtype=float).reshape(5, 3)

    velocity = make_motion_features(sequence, feature_mode="velocity")
    position_velocity = make_motion_features(sequence, feature_mode="position_velocity")

    assert velocity.shape == (4, 3)
    assert position_velocity.shape == (4, 6)
    np.testing.assert_allclose(velocity, np.diff(sequence, axis=0))
    np.testing.assert_allclose(position_velocity[:, :3], sequence[1:])
    np.testing.assert_allclose(position_velocity[:, 3:], velocity)


def test_local_subspace_extraction_is_deterministic():
    rng = np.random.default_rng(456)
    sequence = rng.normal(size=(18, 5))

    first = compute_local_subspace_sequence(
        sequence,
        sequence_id="det",
        rank=2,
        window_length=6,
        stride=3,
    )
    second = compute_local_subspace_sequence(
        sequence,
        sequence_id="det",
        rank=2,
        window_length=6,
        stride=3,
    )

    assert len(first.windows) == len(second.windows)
    for first_window, second_window in zip(first.windows, second.windows):
        np.testing.assert_allclose(first_window.basis, second_window.basis)
        np.testing.assert_allclose(
            first_window.singular_values,
            second_window.singular_values,
        )
