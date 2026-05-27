import numpy as np
import pytest

from src.baselines.mlp import MLPBaseline, pooled_sequence_features


pytest.importorskip("torch")


def _synthetic_sequences() -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(52)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    for label, offset in enumerate([-1.0, 1.0]):
        for index in range(8):
            length = 5 + (index % 3)
            sequences.append(rng.normal(loc=offset, scale=0.1, size=(length, 4)))
            labels.append(label)
    return sequences, np.asarray(labels)


def test_pooled_sequence_features_concatenates_mean_max_std():
    sequences, _ = _synthetic_sequences()

    features = pooled_sequence_features(sequences)

    assert features.shape == (16, 12)


def test_mlp_baseline_fit_predict_on_synthetic_sequences():
    sequences, labels = _synthetic_sequences()
    model = MLPBaseline(
        hidden_dim=16,
        max_epochs=5,
        batch_size=4,
        validation_fraction=0.0,
        seed=52,
    )

    model.fit(sequences, labels)
    predictions = model.predict(sequences[:3])

    assert predictions.shape == (3,)
    assert set(predictions.tolist()).issubset(set(labels.tolist()))
    assert len(model.training_history_) >= 1
