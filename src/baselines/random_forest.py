"""Random Forest baseline over hand-crafted skeleton-sequence features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class RandomForestBaseline:
    """Sklearn-style Random Forest classifier for skeleton sequences."""

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int | None = None,
        seed: int = 0,
        n_jobs: int | None = None,
        max_correlation_units: int = 10,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if max_correlation_units < 2:
            raise ValueError("max_correlation_units must be at least 2.")

        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.seed = int(seed)
        self.n_jobs = n_jobs
        self.max_correlation_units = int(max_correlation_units)

    def fit(self, X_list: Sequence[np.ndarray], y: Sequence[Any]) -> "RandomForestBaseline":
        """Fit the Random Forest classifier and return ``self``."""
        RandomForestClassifier = _load_random_forest_classifier()
        features = handcrafted_sequence_features(
            X_list,
            max_correlation_units=self.max_correlation_units,
        )
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if labels.shape[0] != features.shape[0]:
            raise ValueError("X_list and y must contain the same number of samples.")
        if labels.size == 0:
            raise ValueError("Cannot fit RandomForestBaseline on an empty dataset.")

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        model.fit(features, labels)
        self.model_ = model
        self.classes_ = model.classes_
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Predict labels for a sequence collection."""
        self._check_is_fitted()
        features = handcrafted_sequence_features(
            X_list,
            max_correlation_units=self.max_correlation_units,
        )
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected feature dimension {self.n_features_in_}, "
                f"got {features.shape[1]}."
            )
        return self.model_.predict(features)

    def predict_proba(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Return class probabilities for a sequence collection."""
        self._check_is_fitted()
        features = handcrafted_sequence_features(
            X_list,
            max_correlation_units=self.max_correlation_units,
        )
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected feature dimension {self.n_features_in_}, "
                f"got {features.shape[1]}."
            )
        return self.model_.predict_proba(features)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("RandomForestBaseline must be fitted before prediction.")


def handcrafted_sequence_features(
    X_list: Sequence[np.ndarray],
    *,
    max_correlation_units: int = 10,
) -> np.ndarray:
    """Extract hand-crafted per-sequence features for a collection of sequences."""
    sequences = _validate_sequences(X_list)
    return np.stack(
        [
            _features_for_one_sequence(
                sequence,
                max_correlation_units=max_correlation_units,
            )
            for sequence in sequences
        ],
        axis=0,
    )


def _features_for_one_sequence(
    sequence: np.ndarray,
    *,
    max_correlation_units: int,
) -> np.ndarray:
    velocity_values = _velocity(sequence)
    summary_features = [
        sequence.mean(axis=0),
        sequence.std(axis=0),
        sequence.max(axis=0) - sequence.min(axis=0),
        sequence[0],
        sequence[-1],
        velocity_values.mean(axis=0),
        velocity_values.std(axis=0),
    ]
    return np.concatenate(
        [
            *summary_features,
            _top_unit_pairwise_correlations(
                sequence,
                max_units=max_correlation_units,
            ),
        ]
    ).astype(np.float64, copy=False)


def _velocity(sequence: np.ndarray) -> np.ndarray:
    if sequence.shape[0] < 2:
        return np.zeros((1, sequence.shape[1]), dtype=np.float64)
    return np.diff(sequence, axis=0)


def _top_unit_pairwise_correlations(
    sequence: np.ndarray,
    *,
    max_units: int,
) -> np.ndarray:
    unit_series = _unit_motion_series(sequence)
    n_units = unit_series.shape[1]
    n_selected = min(max_units, n_units)
    target_length = max_units * (max_units - 1) // 2

    if n_units < 2:
        return np.zeros(target_length, dtype=np.float64)

    unit_variance = unit_series.var(axis=0)
    selected = np.argsort(unit_variance)[-n_selected:]
    selected_series = unit_series[:, selected]

    correlations: list[float] = []
    for first in range(n_selected):
        for second in range(first + 1, n_selected):
            correlations.append(
                _safe_correlation(
                    selected_series[:, first],
                    selected_series[:, second],
                )
            )

    if len(correlations) < target_length:
        correlations.extend([0.0] * (target_length - len(correlations)))
    return np.asarray(correlations, dtype=np.float64)


def _unit_motion_series(sequence: np.ndarray) -> np.ndarray:
    feature_dim = int(sequence.shape[1])
    if feature_dim % 3 == 0 and feature_dim >= 3:
        joints = sequence.reshape(sequence.shape[0], feature_dim // 3, 3)
        centered = joints - joints.mean(axis=0, keepdims=True)
        return np.linalg.norm(centered, axis=2)
    return sequence


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def _validate_sequences(X_list: Sequence[np.ndarray]) -> list[np.ndarray]:
    sequences = [np.asarray(sequence, dtype=np.float64) for sequence in X_list]
    if not sequences:
        raise ValueError("X_list must not be empty.")

    feature_dim = sequences[0].shape[1] if sequences[0].ndim == 2 else None
    for sequence in sequences:
        if sequence.ndim != 2:
            raise ValueError("Every sequence must be a 2D T x D matrix.")
        if sequence.shape[0] == 0:
            raise ValueError("Every sequence must contain at least one frame.")
        if feature_dim is None or sequence.shape[1] != feature_dim:
            raise ValueError("Every sequence must have the same feature dimension.")
        if not np.isfinite(sequence).all():
            raise ValueError("Sequences must contain only finite values.")
    return sequences


def _load_random_forest_classifier() -> Any:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "scikit-learn is required for RandomForestBaseline. "
            "Install scikit-learn to use this baseline."
        ) from exc
    return RandomForestClassifier
