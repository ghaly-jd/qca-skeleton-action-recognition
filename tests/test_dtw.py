from __future__ import annotations

import numpy as np
import pytest

from src.distances.dtw import (
    dtw_distance,
    pairwise_dtw_distances,
    window_ratio_label,
)


def test_identical_sequences_have_zero_dtw_distance():
    sequence = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]])

    assert np.isclose(dtw_distance(sequence, sequence), 0.0)


def test_path_length_normalization_divides_accumulated_cost():
    sequence_x = np.asarray([[0.0], [2.0]])
    sequence_y = np.asarray([[0.0], [1.0], [2.0]])

    raw_distance = dtw_distance(
        sequence_x,
        sequence_y,
        normalize_by_path_length=False,
    )
    normalized_distance = dtw_distance(
        sequence_x,
        sequence_y,
        normalize_by_path_length=True,
    )

    assert np.isclose(raw_distance, 1.0)
    assert np.isclose(normalized_distance, 1.0 / 3.0)


def test_window_ratio_is_widened_for_different_sequence_lengths():
    sequence_x = np.asarray([[0.0], [1.0]])
    sequence_y = np.asarray([[0.0], [0.5], [1.0], [1.5]])

    constrained = dtw_distance(sequence_x, sequence_y, window_ratio=0.0)
    unconstrained = dtw_distance(sequence_x, sequence_y, window_ratio=None)

    assert np.isfinite(constrained)
    assert constrained >= 0.0
    assert unconstrained >= 0.0


def test_pairwise_dtw_distances_shape_and_values():
    train = [
        np.asarray([[0.0], [1.0], [2.0]]),
        np.asarray([[10.0], [11.0], [12.0]]),
    ]
    test = [np.asarray([[0.0], [1.0], [2.0]])]

    distances = pairwise_dtw_distances(test, train)

    assert distances.shape == (1, 2)
    assert np.isclose(distances[0, 0], 0.0)
    assert distances[0, 1] > distances[0, 0]


def test_pairwise_dtw_matches_scalar_dtw():
    train = [
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]]),
        np.asarray([[0.0, 1.0], [1.0, 1.0]]),
    ]
    test = [
        np.asarray([[0.0, 0.0], [1.5, 0.5], [2.0, 1.0]]),
        np.asarray([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0], [1.5, 1.0]]),
    ]

    distances = pairwise_dtw_distances(test, train, window_ratio=0.25)
    expected = np.asarray(
        [
            [
                dtw_distance(test_sequence, train_sequence, window_ratio=0.25)
                for train_sequence in train
            ]
            for test_sequence in test
        ]
    )

    assert np.allclose(distances, expected)


def test_torch_cpu_dtw_matches_scalar_dtw():
    pytest.importorskip("torch")
    train = [
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]]),
        np.asarray([[0.0, 1.0], [1.0, 1.0]]),
    ]
    test = [
        np.asarray([[0.0, 0.0], [1.5, 0.5], [2.0, 1.0]]),
        np.asarray([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0], [1.5, 1.0]]),
    ]

    distances = pairwise_dtw_distances(
        test,
        train,
        window_ratio=0.25,
        backend="torch_cpu",
        torch_dtype="float64",
    )
    expected = np.asarray(
        [
            [
                dtw_distance(test_sequence, train_sequence, window_ratio=0.25)
                for train_sequence in train
            ]
            for test_sequence in test
        ]
    )

    assert np.allclose(distances, expected)


def test_window_ratio_label_is_stable_for_csv_output():
    assert window_ratio_label(None) == "none"
    assert window_ratio_label("none") == "none"
    assert window_ratio_label("0.10") == "0.1"
