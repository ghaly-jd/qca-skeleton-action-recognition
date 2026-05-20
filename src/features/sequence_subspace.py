"""Per-sequence SVD subspace extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.features.normalization import normalize_sequence_matrix


@dataclass(frozen=True)
class SequenceSubspace:
    """Low-dimensional motion subspace for one sequence."""

    sequence_id: str
    basis: np.ndarray
    singular_values: np.ndarray
    num_frames: int
    feature_dim: int
    rank: int


def compute_sequence_subspace(
    matrix: np.ndarray,
    *,
    rank: int,
    sequence_id: str = "",
    center_sequence: bool = True,
    min_frames_required: int = 2,
    feature_standardization: str = "none",
    frame_l2_normalization: bool = False,
    check_orthonormal: bool = True,
) -> SequenceSubspace:
    """Compute the top ``rank`` right singular vectors as a ``D x rank`` basis."""
    if rank <= 0:
        raise ValueError("rank must be positive.")

    values = normalize_sequence_matrix(
        matrix,
        sequence_centering=center_sequence,
        feature_standardization=feature_standardization,
        frame_l2_normalization=frame_l2_normalization,
    )
    num_frames, feature_dim = values.shape
    if num_frames < min_frames_required:
        raise ValueError(
            f"{sequence_id or 'sequence'} has {num_frames} frames, "
            f"below min_frames_required={min_frames_required}."
        )
    if num_frames <= rank:
        raise ValueError(
            f"{sequence_id or 'sequence'} has {num_frames} frames; "
            f"need at least rank + 1 frames for centered rank {rank}."
        )
    if feature_dim < rank:
        raise ValueError(
            f"{sequence_id or 'sequence'} has feature_dim={feature_dim}, "
            f"below rank={rank}."
        )

    _, singular_values, vh = np.linalg.svd(values, full_matrices=False)
    if vh.shape[0] < rank:
        raise ValueError(
            f"{sequence_id or 'sequence'} SVD returned only {vh.shape[0]} vectors "
            f"for rank={rank}."
        )

    basis = vh[:rank].T
    if check_orthonormal and not is_orthonormal(basis):
        raise ValueError(f"{sequence_id or 'sequence'} basis is not orthonormal.")

    return SequenceSubspace(
        sequence_id=sequence_id,
        basis=basis,
        singular_values=singular_values[:rank],
        num_frames=int(num_frames),
        feature_dim=int(feature_dim),
        rank=int(rank),
    )


def compute_subspaces(
    sequences: Iterable[np.ndarray],
    sequence_ids: Iterable[str],
    *,
    rank: int,
    center_sequence: bool = True,
    min_frames_required: int = 2,
    feature_standardization: str = "none",
    frame_l2_normalization: bool = False,
) -> list[SequenceSubspace]:
    """Compute subspaces for a sequence collection."""
    return [
        compute_sequence_subspace(
            sequence,
            rank=rank,
            sequence_id=sequence_id,
            center_sequence=center_sequence,
            min_frames_required=min_frames_required,
            feature_standardization=feature_standardization,
            frame_l2_normalization=frame_l2_normalization,
        )
        for sequence, sequence_id in zip(sequences, sequence_ids)
    ]


def stack_bases(subspaces: Iterable[SequenceSubspace]) -> np.ndarray:
    """Stack ``D x r`` basis matrices into ``N x D x r`` form."""
    bases = [subspace.basis for subspace in subspaces]
    if not bases:
        raise ValueError("No subspaces to stack.")
    return np.stack(bases, axis=0)


def is_orthonormal(basis: np.ndarray, *, atol: float = 1e-8) -> bool:
    """Return whether basis columns are orthonormal."""
    values = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2:
        return False
    gram = values.T @ values
    return bool(np.allclose(gram, np.eye(values.shape[1]), atol=atol))
