"""Amplitude-encoding helpers for real skeleton feature vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AmplitudeEncodedState:
    """A normalized vector ready to be loaded as quantum amplitudes."""

    amplitudes: np.ndarray
    original_dim: int
    padded_dim: int
    num_qubits: int
    norm: float


def amplitude_encode(
    vector: np.ndarray,
    *,
    target_dim: int | None = None,
) -> AmplitudeEncodedState:
    """Return a normalized, power-of-two padded amplitude vector."""
    values = _as_vector(vector)
    original_dim = int(values.shape[0])
    padded_dim = int(target_dim or next_power_of_two(original_dim))
    if padded_dim < original_dim:
        raise ValueError(
            f"target_dim={padded_dim} cannot be smaller than vector length {original_dim}."
        )
    if not is_power_of_two(padded_dim):
        raise ValueError("target_dim must be a power of two.")

    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("Cannot amplitude-encode a zero vector.")

    amplitudes = np.zeros(padded_dim, dtype=np.complex128)
    amplitudes[:original_dim] = values.astype(np.complex128) / norm
    return AmplitudeEncodedState(
        amplitudes=amplitudes,
        original_dim=original_dim,
        padded_dim=padded_dim,
        num_qubits=int(np.log2(padded_dim)),
        norm=norm,
    )


def amplitude_encode_pair(
    vector_x: np.ndarray,
    vector_y: np.ndarray,
) -> tuple[AmplitudeEncodedState, AmplitudeEncodedState]:
    """Amplitude-encode two vectors into the same Hilbert-space dimension."""
    x = _as_vector(vector_x)
    y = _as_vector(vector_y)
    padded_dim = next_power_of_two(max(x.shape[0], y.shape[0]))
    return (
        amplitude_encode(x, target_dim=padded_dim),
        amplitude_encode(y, target_dim=padded_dim),
    )


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    if value <= 0:
        raise ValueError("value must be positive.")
    return 1 << (int(value) - 1).bit_length()


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and (value & (value - 1)) == 0


def _as_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional vector.")
    if values.shape[0] == 0:
        raise ValueError("Cannot amplitude-encode an empty vector.")
    if not np.isfinite(values).all():
        raise ValueError("Amplitude-encoding input contains non-finite values.")
    return values
