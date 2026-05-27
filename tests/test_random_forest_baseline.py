import numpy as np
import pytest

from src.baselines.random_forest import (
    RandomForestBaseline,
    handcrafted_sequence_features,
)


pytest.importorskip("sklearn")


def _synthetic_sequences() -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(54)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    for label, offset in enumerate([-1.0, 1.0]):
        for index in range(8):
            length = 5 + (index % 4)
            sequences.append(rng.normal(loc=offset, scale=0.1, size=(length, 6)))
            labels.append(label)
    return sequences, np.asarray(labels)


def test_handcrafted_sequence_features_have_expected_shape():
    sequences, _ = _synthetic_sequences()

    features = handcrafted_sequence_features(sequences, max_correlation_units=4)

    assert features.shape == (16, 48)


def test_random_forest_baseline_fit_predict_on_synthetic_sequences():
    sequences, labels = _synthetic_sequences()
    model = RandomForestBaseline(
        n_estimators=10,
        seed=54,
        max_correlation_units=4,
    )

    model.fit(sequences, labels)
    predictions = model.predict(sequences[:4])
    probabilities = model.predict_proba(sequences[:4])

    assert predictions.shape == (4,)
    assert probabilities.shape == (4, 2)
    assert set(predictions.tolist()).issubset(set(labels.tolist()))
