"""Train-split PCA projection for frame-wise skeleton features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAModel:
    """Frame-wise PCA model fitted only on training frames."""

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.components.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.components.shape[0])

    def transform_sequence(self, sequence: np.ndarray) -> np.ndarray:
        values = _as_2d_float(sequence)
        if values.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {values.shape[1]}."
            )
        return (values - self.mean) @ self.components

    def transform_sequences(self, sequences: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [self.transform_sequence(sequence) for sequence in sequences]


def fit_pca_from_sequences(
    sequences: Sequence[np.ndarray],
    *,
    n_components: int,
) -> PCAModel:
    """Fit PCA on concatenated training frames."""
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    frames = _stack_frames(sequences)
    num_frames, feature_dim = frames.shape
    if n_components > feature_dim:
        raise ValueError(
            f"n_components={n_components} exceeds feature_dim={feature_dim}."
        )
    if n_components > num_frames:
        raise ValueError(
            f"n_components={n_components} exceeds number of training frames={num_frames}."
        )

    mean = frames.mean(axis=0)
    centered = frames - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < n_components:
        raise ValueError(
            f"PCA SVD returned only {vh.shape[0]} components for k={n_components}."
        )

    selected_singular_values = singular_values[:n_components]
    total_variance = float(np.sum(singular_values**2))
    if total_variance > 0.0:
        explained = (selected_singular_values**2) / total_variance
    else:
        explained = np.zeros_like(selected_singular_values)

    return PCAModel(
        mean=mean,
        components=vh[:n_components].T,
        singular_values=selected_singular_values,
        explained_variance_ratio=explained,
    )


def transform_sequences(
    sequences: Sequence[np.ndarray],
    model: PCAModel,
) -> list[np.ndarray]:
    """Transform a sequence collection using a fitted PCA model."""
    return model.transform_sequences(sequences)


def _stack_frames(sequences: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [_as_2d_float(sequence) for sequence in sequences]
    if not arrays:
        raise ValueError("At least one sequence is required to fit PCA.")

    feature_dim = arrays[0].shape[1]
    for sequence in arrays:
        if sequence.shape[0] == 0:
            raise ValueError("Sequences must have at least one frame.")
        if sequence.shape[1] != feature_dim:
            raise ValueError("All sequences must have the same feature dimension.")
    return np.vstack(arrays)


def _as_2d_float(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a 2D T x D sequence matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Sequences must contain only finite values.")
    return values
