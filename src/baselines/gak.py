"""Global Alignment Kernel baseline following Cuturi (2011)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class GAKBaseline:
    """1-NN classifier using normalized Global Alignment Kernel similarity."""

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        normalize: bool = True,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        self.sigma = float(sigma)
        self.normalize = bool(normalize)

    def fit(self, X_list: Sequence[np.ndarray], y: Sequence[Any]) -> "GAKBaseline":
        """Store training sequences and labels for 1-NN prediction."""
        sequences = _validate_sequences(X_list)
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if labels.shape[0] != len(sequences):
            raise ValueError("X_list and y must contain the same number of samples.")
        if labels.size == 0:
            raise ValueError("Cannot fit GAKBaseline on an empty dataset.")

        self.train_sequences_ = sequences
        self.train_labels_ = labels
        self.train_self_kernels_ = np.asarray(
            [
                gak_kernel(
                    sequence,
                    sequence,
                    sigma=self.sigma,
                    normalize=False,
                )
                for sequence in sequences
            ],
            dtype=np.float64,
        )
        return self

    def predict(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Predict labels by nearest training sequence in GAK similarity."""
        similarities = self.similarity_matrix(X_list)
        nearest_indices = np.argmax(similarities, axis=1)
        return self.train_labels_[nearest_indices]

    def similarity_matrix(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Return normalized GAK similarities with shape ``N_test x N_train``."""
        self._check_is_fitted()
        test_sequences = _validate_sequences(X_list)
        similarities = np.empty(
            (len(test_sequences), len(self.train_sequences_)),
            dtype=np.float64,
        )
        for test_index, test_sequence in enumerate(test_sequences):
            test_self_kernel = gak_kernel(
                test_sequence,
                test_sequence,
                sigma=self.sigma,
                normalize=False,
            )
            for train_index, train_sequence in enumerate(self.train_sequences_):
                raw_kernel = gak_kernel(
                    test_sequence,
                    train_sequence,
                    sigma=self.sigma,
                    normalize=False,
                )
                if self.normalize:
                    similarities[test_index, train_index] = _normalize_kernel(
                        raw_kernel,
                        test_self_kernel,
                        self.train_self_kernels_[train_index],
                    )
                else:
                    similarities[test_index, train_index] = raw_kernel
        return similarities

    def distance_matrix(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Return kernel distances with shape ``N_test x N_train``."""
        similarities = self.similarity_matrix(X_list)
        if self.normalize:
            return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarities))

        self._check_is_fitted()
        test_sequences = _validate_sequences(X_list)
        distances = np.empty_like(similarities)
        for test_index, test_sequence in enumerate(test_sequences):
            test_self_kernel = gak_kernel(
                test_sequence,
                test_sequence,
                sigma=self.sigma,
                normalize=False,
            )
            values = (
                test_self_kernel
                + self.train_self_kernels_
                - (2.0 * similarities[test_index])
            )
            distances[test_index] = np.sqrt(np.maximum(0.0, values))
        return distances

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "train_sequences_"):
            raise RuntimeError("GAKBaseline must be fitted before prediction.")


def gak_kernel(
    sequence_x: np.ndarray,
    sequence_y: np.ndarray,
    *,
    sigma: float = 1.0,
    normalize: bool = True,
) -> float:
    """Return Global Alignment Kernel similarity for two ``T x D`` sequences."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    x, y = _validate_pair(sequence_x, sequence_y)
    raw_kernel = _raw_gak_kernel(x, y, sigma=float(sigma))
    if not normalize:
        return raw_kernel

    self_x = _raw_gak_kernel(x, x, sigma=float(sigma))
    self_y = _raw_gak_kernel(y, y, sigma=float(sigma))
    return _normalize_kernel(raw_kernel, self_x, self_y)


def pairwise_gak_kernel(
    test_sequences: Sequence[np.ndarray],
    train_sequences: Sequence[np.ndarray],
    *,
    sigma: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Return pairwise GAK similarities with shape ``N_test x N_train``."""
    test = _validate_sequences(test_sequences)
    train = _validate_sequences(train_sequences)

    train_self = None
    test_self = None
    if normalize:
        train_self = np.asarray(
            [
                gak_kernel(sequence, sequence, sigma=sigma, normalize=False)
                for sequence in train
            ],
            dtype=np.float64,
        )
        test_self = np.asarray(
            [
                gak_kernel(sequence, sequence, sigma=sigma, normalize=False)
                for sequence in test
            ],
            dtype=np.float64,
        )

    similarities = np.empty((len(test), len(train)), dtype=np.float64)
    for test_index, test_sequence in enumerate(test):
        for train_index, train_sequence in enumerate(train):
            raw_kernel = gak_kernel(
                test_sequence,
                train_sequence,
                sigma=sigma,
                normalize=False,
            )
            if normalize:
                similarities[test_index, train_index] = _normalize_kernel(
                    raw_kernel,
                    test_self[test_index],
                    train_self[train_index],
                )
            else:
                similarities[test_index, train_index] = raw_kernel
    return similarities


def _raw_gak_kernel(
    sequence_x: np.ndarray,
    sequence_y: np.ndarray,
    *,
    sigma: float,
) -> float:
    local_kernel = _local_gak_kernel_matrix(sequence_x, sequence_y, sigma=sigma)
    n_frames_x, n_frames_y = local_kernel.shape
    scores = np.zeros((n_frames_x + 1, n_frames_y + 1), dtype=np.float64)

    for i in range(1, n_frames_x + 1):
        for j in range(1, n_frames_y + 1):
            predecessor_sum = (
                1.0
                + scores[i - 1, j]
                + scores[i, j - 1]
                + scores[i - 1, j - 1]
            )
            scores[i, j] = local_kernel[i - 1, j - 1] * predecessor_sum
    return float(scores[n_frames_x, n_frames_y])


def _local_gak_kernel_matrix(
    sequence_x: np.ndarray,
    sequence_y: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    differences = sequence_x[:, None, :] - sequence_y[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    base = np.exp(-squared_distances / (2.0 * sigma * sigma))
    return (base / (2.0 - base)).astype(np.float64, copy=False)


def _normalize_kernel(raw_kernel: float, self_x: float, self_y: float) -> float:
    denominator = np.sqrt(max(float(self_x), 0.0) * max(float(self_y), 0.0))
    if denominator == 0.0:
        return 0.0
    normalized = float(raw_kernel) / denominator
    return float(np.clip(normalized, 0.0, 1.0))


def _validate_pair(
    sequence_x: np.ndarray,
    sequence_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sequence_x, dtype=np.float64)
    y = np.asarray(sequence_y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("GAK inputs must be 2D T x D matrices.")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("GAK inputs must have at least one frame.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("GAK inputs must have the same feature dimension.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("GAK inputs must contain only finite values.")
    return x, y


def _validate_sequences(sequences: Sequence[np.ndarray]) -> list[np.ndarray]:
    arrays = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
    if not arrays:
        raise ValueError("sequences must not be empty.")

    feature_dim = arrays[0].shape[1] if arrays[0].ndim == 2 else None
    for sequence in arrays:
        if sequence.ndim != 2:
            raise ValueError("Every sequence must be a 2D T x D matrix.")
        if sequence.shape[0] == 0:
            raise ValueError("Every sequence must have at least one frame.")
        if feature_dim is None or sequence.shape[1] != feature_dim:
            raise ValueError("Every sequence must have the same feature dimension.")
        if not np.isfinite(sequence).all():
            raise ValueError("Sequences must contain only finite values.")
    return arrays
