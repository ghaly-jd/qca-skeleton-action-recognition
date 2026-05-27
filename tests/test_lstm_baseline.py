import numpy as np
import pytest

from src.baselines.lstm import LSTMBaseline


pytest.importorskip("torch")


def _synthetic_sequences() -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(53)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    for label, offset in enumerate([-0.75, 0.75]):
        for index in range(8):
            length = 4 + (index % 4)
            trend = np.linspace(0.0, offset, num=length)[:, None]
            noise = rng.normal(scale=0.05, size=(length, 3))
            sequences.append(np.repeat(trend, repeats=3, axis=1) + noise)
            labels.append(label)
    return sequences, np.asarray(labels)


def test_lstm_baseline_fit_predict_on_synthetic_sequences():
    sequences, labels = _synthetic_sequences()
    model = LSTMBaseline(
        hidden_dim=8,
        max_epochs=3,
        batch_size=4,
        validation_fraction=0.0,
        seed=53,
    )

    model.fit(sequences, labels)
    predictions = model.predict(sequences[:4])
    probabilities = model.predict_proba(sequences[:4])

    assert predictions.shape == (4,)
    assert probabilities.shape == (4, 2)
    assert set(predictions.tolist()).issubset(set(labels.tolist()))
    assert len(model.training_history_) >= 1
