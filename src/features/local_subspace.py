"""Local motion-subspace extraction for skeleton sequences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.features.normalization import normalize_sequence_matrix
from src.features.sequence_subspace import is_orthonormal


FEATURE_MODES = ["position", "velocity", "position_velocity"]
SHORT_SEQUENCE_MODES = ["single", "error"]


@dataclass(frozen=True)
class LocalSubspaceWindow:
    """One low-rank motion subspace extracted from a temporal window."""

    sequence_id: str
    window_index: int
    start_frame: int
    end_frame: int
    basis: np.ndarray
    singular_values: np.ndarray
    num_frames: int
    feature_dim: int
    rank: int


@dataclass(frozen=True)
class LocalSubspaceSequence:
    """A skeleton sequence represented as ordered local motion subspaces."""

    sequence_id: str
    windows: list[LocalSubspaceWindow]
    window_length: int
    stride: int
    rank: int
    feature_mode: str


def make_sliding_windows(
    sequence: np.ndarray,
    *,
    window_length: int,
    stride: int,
    short_sequence_mode: str = "single",
    include_tail: bool = True,
) -> list[tuple[int, int, np.ndarray]]:
    """Return deterministic sliding windows as ``(start, end, values)`` tuples."""
    values = _as_sequence_matrix(sequence)
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    short_sequence_mode = short_sequence_mode.lower()
    if short_sequence_mode not in SHORT_SEQUENCE_MODES:
        known = ", ".join(SHORT_SEQUENCE_MODES)
        raise ValueError(f"Unknown short_sequence_mode '{short_sequence_mode}': {known}.")

    num_frames = values.shape[0]
    if num_frames < window_length:
        if short_sequence_mode == "error":
            raise ValueError(
                f"Sequence has {num_frames} frames, below window_length={window_length}."
            )
        return [(0, num_frames, values.copy())]

    starts = list(range(0, num_frames - window_length + 1, stride))
    if include_tail and starts:
        last_end = starts[-1] + window_length
        missing_frames = num_frames - last_end
        tail_start = num_frames - window_length
        if missing_frames > stride / 2 and tail_start != starts[-1]:
            starts.append(tail_start)

    return [
        (int(start), int(start + window_length), values[start : start + window_length].copy())
        for start in starts
    ]


def compute_local_subspace_sequence(
    sequence: np.ndarray,
    *,
    sequence_id: str = "",
    rank: int,
    window_length: int,
    stride: int,
    center_sequence: bool = True,
    feature_standardization: str = "none",
    frame_l2_normalization: bool = False,
    feature_mode: str = "position",
    short_sequence_mode: str = "single",
    check_orthonormal: bool = True,
) -> LocalSubspaceSequence:
    """Convert one ``T x D`` skeleton sequence into local subspace windows."""
    if rank <= 0:
        raise ValueError("rank must be positive.")

    feature_mode = feature_mode.lower()
    if feature_mode not in FEATURE_MODES:
        known = ", ".join(FEATURE_MODES)
        raise ValueError(f"Unknown feature_mode '{feature_mode}': {known}.")

    feature_values = make_motion_features(sequence, feature_mode=feature_mode)
    feature_dim = feature_values.shape[1]
    if feature_dim < rank:
        raise ValueError(
            f"{sequence_id or 'sequence'} has feature_dim={feature_dim}, below rank={rank}."
        )

    windows = make_sliding_windows(
        feature_values,
        window_length=window_length,
        stride=stride,
        short_sequence_mode=short_sequence_mode,
    )
    local_windows: list[LocalSubspaceWindow] = []
    for window_index, (start_frame, end_frame, window_values) in enumerate(windows):
        if window_values.shape[0] <= rank:
            raise ValueError(
                f"{sequence_id or 'sequence'} window {window_index} has "
                f"{window_values.shape[0]} frames; need more than rank={rank}."
            )
        basis, singular_values = _compute_window_basis(
            window_values,
            rank=rank,
            center_sequence=center_sequence,
            feature_standardization=feature_standardization,
            frame_l2_normalization=frame_l2_normalization,
            check_orthonormal=check_orthonormal,
            sequence_id=sequence_id,
            window_index=window_index,
        )
        local_windows.append(
            LocalSubspaceWindow(
                sequence_id=sequence_id,
                window_index=window_index,
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                basis=basis,
                singular_values=singular_values,
                num_frames=int(window_values.shape[0]),
                feature_dim=int(feature_dim),
                rank=int(rank),
            )
        )

    if not local_windows:
        raise ValueError(f"{sequence_id or 'sequence'} produced no local windows.")

    return LocalSubspaceSequence(
        sequence_id=sequence_id,
        windows=local_windows,
        window_length=int(window_length),
        stride=int(stride),
        rank=int(rank),
        feature_mode=feature_mode,
    )


def compute_local_subspace_sequences(
    sequences: Sequence[np.ndarray],
    sequence_ids: Sequence[str],
    *,
    rank: int,
    window_length: int,
    stride: int,
    center_sequence: bool = True,
    feature_standardization: str = "none",
    frame_l2_normalization: bool = False,
    feature_mode: str = "position",
    short_sequence_mode: str = "single",
) -> list[LocalSubspaceSequence]:
    """Compute local subspace sequences for a collection of skeleton sequences."""
    if len(sequences) != len(sequence_ids):
        raise ValueError("sequences and sequence_ids must have the same length.")
    return [
        compute_local_subspace_sequence(
            sequence,
            sequence_id=sequence_id,
            rank=rank,
            window_length=window_length,
            stride=stride,
            center_sequence=center_sequence,
            feature_standardization=feature_standardization,
            frame_l2_normalization=frame_l2_normalization,
            feature_mode=feature_mode,
            short_sequence_mode=short_sequence_mode,
        )
        for sequence, sequence_id in zip(sequences, sequence_ids)
    ]


def local_bases(local_sequence: LocalSubspaceSequence) -> list[np.ndarray]:
    """Return the ordered basis matrices from a local subspace sequence."""
    return [window.basis for window in local_sequence.windows]


def num_local_windows(local_sequence: LocalSubspaceSequence) -> int:
    """Return the number of local subspace windows."""
    return len(local_sequence.windows)


def make_motion_features(
    sequence: np.ndarray,
    *,
    feature_mode: str = "position",
) -> np.ndarray:
    """Return position, velocity, or concatenated position/velocity features."""
    values = _as_sequence_matrix(sequence)
    feature_mode = feature_mode.lower()
    if feature_mode == "position":
        return values.copy()
    if feature_mode == "velocity":
        if values.shape[0] < 2:
            raise ValueError("velocity features require at least two frames.")
        return np.diff(values, axis=0)
    if feature_mode == "position_velocity":
        if values.shape[0] < 2:
            raise ValueError("position_velocity features require at least two frames.")
        velocity = np.diff(values, axis=0)
        return np.concatenate([values[1:], velocity], axis=1)

    known = ", ".join(FEATURE_MODES)
    raise ValueError(f"Unknown feature_mode '{feature_mode}': {known}.")


def _compute_window_basis(
    matrix: np.ndarray,
    *,
    rank: int,
    center_sequence: bool,
    feature_standardization: str,
    frame_l2_normalization: bool,
    check_orthonormal: bool,
    sequence_id: str,
    window_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = normalize_sequence_matrix(
        matrix,
        sequence_centering=center_sequence,
        feature_standardization=feature_standardization,
        frame_l2_normalization=frame_l2_normalization,
    )
    _, singular_values, vh = np.linalg.svd(values, full_matrices=False)
    if vh.shape[0] < rank:
        raise ValueError(
            f"{sequence_id or 'sequence'} window {window_index} SVD returned only "
            f"{vh.shape[0]} vectors for rank={rank}."
        )
    basis = vh[:rank].T
    if check_orthonormal and not is_orthonormal(basis):
        raise ValueError(
            f"{sequence_id or 'sequence'} window {window_index} basis is not orthonormal."
        )
    return basis, singular_values[:rank]


def _as_sequence_matrix(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("sequence must be a 2D T x D matrix.")
    if values.shape[0] == 0:
        raise ValueError("sequence must have at least one frame.")
    if values.shape[1] == 0:
        raise ValueError("sequence must have at least one feature.")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values.")
    return values
