from __future__ import annotations

import numpy as np
import pytest

from src.features.pca_projection import fit_pca_from_sequences, transform_sequences


def test_pca_projection_fits_on_training_frames_and_transforms_sequences():
    train = [
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]),
    ]
    test = [np.asarray([[0.5, 1.0, 0.0], [1.5, 1.0, 0.0]])]

    model = fit_pca_from_sequences(train, n_components=2)
    train_projected = transform_sequences(train, model)
    test_projected = transform_sequences(test, model)

    assert model.mean.shape == (3,)
    assert model.components.shape == (3, 2)
    assert train_projected[0].shape == (2, 2)
    assert test_projected[0].shape == (2, 2)


def test_pca_components_are_orthonormal():
    rng = np.random.default_rng(0)
    train = [rng.normal(size=(8, 5)), rng.normal(size=(7, 5))]

    model = fit_pca_from_sequences(train, n_components=3)

    assert np.allclose(model.components.T @ model.components, np.eye(3))


def test_pca_rejects_more_components_than_features():
    train = [np.ones((4, 3))]

    with pytest.raises(ValueError, match="exceeds feature_dim"):
        fit_pca_from_sequences(train, n_components=4)
