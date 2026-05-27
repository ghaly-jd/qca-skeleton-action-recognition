import numpy as np
import pytest

from src.baselines.kdtw import (
    KDTWBaseline,
    kdtw_kernel,
    pairwise_kdtw_kernel,
)


def test_normalized_kdtw_self_similarity_is_one():
    sequence = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    similarity = kdtw_kernel(sequence, sequence, gamma=1.0, normalize=True)

    assert similarity == pytest.approx(1.0, abs=1e-12)


def test_normalized_kdtw_orthogonal_constant_sequences_are_low():
    first = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (5, 1))
    second = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float64), (5, 1))

    similarity = kdtw_kernel(first, second, gamma=5.0, normalize=True)

    assert 0.0 <= similarity < 1e-3


def test_pairwise_kdtw_kernel_shape_and_diagonal():
    first = np.eye(3, dtype=np.float64)
    second = np.flipud(first)
    sequences = [first, second]

    similarities = pairwise_kdtw_kernel(sequences, sequences, gamma=1.0)

    assert similarities.shape == (2, 2)
    np.testing.assert_allclose(np.diag(similarities), 1.0, atol=1e-12)


def test_kdtw_baseline_predicts_nearest_synthetic_sequences():
    train_sequences = [
        np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (4, 1)),
        np.tile(np.asarray([[0.0, 1.0]], dtype=np.float64), (4, 1)),
    ]
    train_labels = np.asarray(["x", "y"])
    test_sequences = [
        np.tile(np.asarray([[0.95, 0.05]], dtype=np.float64), (4, 1)),
        np.tile(np.asarray([[0.05, 0.95]], dtype=np.float64), (4, 1)),
    ]

    model = KDTWBaseline(gamma=5.0)
    model.fit(train_sequences, train_labels)

    predictions = model.predict(test_sequences)

    np.testing.assert_array_equal(predictions, train_labels)
