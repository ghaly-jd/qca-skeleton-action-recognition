"""Normalization helpers for skeleton sequence feature matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def center_sequence(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a sequence-centered copy of ``matrix`` and the feature mean."""
    values = _as_2d_float(matrix)
    mean = values.mean(axis=0, keepdims=True)
    return values - mean, mean.squeeze(axis=0)


def l2_normalize_frames(matrix: np.ndarray) -> np.ndarray:
    """Normalize each frame vector to unit L2 norm."""
    values = _as_2d_float(matrix)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(values.dtype).eps)


@dataclass(frozen=True)
class FeatureStandardizer:
    """Feature-wise z-score standardizer with explicit fit/transform steps."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "FeatureStandardizer":
        values = _as_2d_float(matrix)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.maximum(scale, np.finfo(values.dtype).eps)
        return cls(mean=mean, scale=scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = _as_2d_float(matrix)
        return (values - self.mean) / self.scale


def standardize_sequence_features(matrix: np.ndarray) -> tuple[np.ndarray, FeatureStandardizer]:
    """Fit and apply per-sequence feature standardization."""
    standardizer = FeatureStandardizer.fit(matrix)
    return standardizer.transform(matrix), standardizer


def normalize_sequence_matrix(
    matrix: np.ndarray,
    *,
    sequence_centering: bool = True,
    feature_standardization: str = "none",
    frame_l2_normalization: bool = False,
) -> np.ndarray:
    """Apply the Phase 2 sequence-level normalization choices."""
    values = _as_2d_float(matrix)
    if sequence_centering:
        values, _ = center_sequence(values)
    if feature_standardization.lower() in {"zscore", "sequence", "per_sequence"}:
        values, _ = standardize_sequence_features(values)
    elif feature_standardization.lower() not in {"", "none", "false"}:
        raise ValueError(f"Unsupported feature_standardization: {feature_standardization}")
    if frame_l2_normalization:
        values = l2_normalize_frames(values)
    return values


def _as_2d_float(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a 2D T x D feature matrix.")
    return values
