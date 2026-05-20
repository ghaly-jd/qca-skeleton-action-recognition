"""Skeleton preprocessing for sequence-as-subspace experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


ROOT_JOINTS_20 = {
    "hip_or_spine": 6,
    "hip": 6,
    "hip_center": 6,
    "spine": 3,
    "shoulder_center": 2,
}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configurable preprocessing choices for one skeleton sequence."""

    remove_invalid_frames: bool = True
    root_center: bool = True
    root_joint: str | int = "hip_or_spine"
    scale_normalize: bool = True
    feature_standardization: str = "none"
    l2_normalize_frames: bool = False
    sequence_centering: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "PreprocessingConfig":
        return cls(**(data or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preprocess_skeleton_sequence(
    joints: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert ``T x J x 3`` joints into a preprocessed ``T x 60`` matrix."""
    config = config or PreprocessingConfig()
    coords = np.asarray(joints, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("joints must have shape T x J x 3.")

    original_frames = int(coords.shape[0])
    valid_mask = np.ones(original_frames, dtype=bool)
    if config.remove_invalid_frames:
        valid_mask = _valid_frame_mask(coords, confidence)
        coords = coords[valid_mask]
        if confidence is not None:
            confidence = np.asarray(confidence)[valid_mask]

    if coords.shape[0] == 0:
        raise ValueError("No valid skeleton frames remain after preprocessing.")

    root_index = resolve_root_joint(config.root_joint, coords.shape[1])
    scale_factor = 1.0

    if config.root_center:
        coords = coords - coords[:, root_index : root_index + 1, :]

    if config.scale_normalize:
        scale_factor = _sequence_scale(coords, root_index=root_index)
        coords = coords / scale_factor

    matrix = coords.reshape(coords.shape[0], coords.shape[1] * coords.shape[2])

    if config.sequence_centering:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)

    matrix = _standardize_features(matrix, config.feature_standardization)

    if config.l2_normalize_frames:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, np.finfo(matrix.dtype).eps)

    info = {
        "original_frames": original_frames,
        "num_frames": int(matrix.shape[0]),
        "removed_frames": int(original_frames - matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "root_joint_index_zero_based": int(root_index),
        "scale_factor": float(scale_factor),
        "preprocessing": config.to_dict(),
    }
    return matrix.astype(np.float32), info


def resolve_root_joint(root_joint: str | int, num_joints: int) -> int:
    """Resolve a root-joint config value to a zero-based joint index."""
    if isinstance(root_joint, int):
        if root_joint == 0:
            return 0
        if 1 <= root_joint <= num_joints:
            return root_joint - 1
        raise ValueError(f"Root joint {root_joint} is outside 1..{num_joints}.")

    key = str(root_joint).strip().lower()
    if key not in ROOT_JOINTS_20:
        known = ", ".join(sorted(ROOT_JOINTS_20))
        raise ValueError(f"Unknown root joint '{root_joint}'. Known values: {known}.")

    index = ROOT_JOINTS_20[key]
    if index >= num_joints:
        raise ValueError(f"Root joint '{root_joint}' is not valid for {num_joints} joints.")
    return index


def _valid_frame_mask(
    coords: np.ndarray,
    confidence: np.ndarray | None,
) -> np.ndarray:
    finite = np.isfinite(coords).all(axis=(1, 2))
    nonzero = np.any(np.abs(coords) > np.finfo(coords.dtype).eps, axis=(1, 2))
    valid = finite & nonzero
    if confidence is not None:
        confidence = np.asarray(confidence)
        valid &= np.isfinite(confidence).all(axis=1)
    return valid


def _sequence_scale(coords: np.ndarray, *, root_index: int) -> float:
    root = coords[:, root_index : root_index + 1, :]
    distances = np.linalg.norm(coords - root, axis=2)
    positive = distances[distances > np.finfo(coords.dtype).eps]
    if positive.size == 0:
        return 1.0
    scale = float(np.median(positive))
    return scale if scale > np.finfo(coords.dtype).eps else 1.0


def _standardize_features(matrix: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode in {"", "none", "false"}:
        return matrix
    if mode in {"sequence", "per_sequence"}:
        std = matrix.std(axis=0, keepdims=True)
        return (matrix - matrix.mean(axis=0, keepdims=True)) / np.maximum(
            std,
            np.finfo(matrix.dtype).eps,
        )
    raise ValueError(
        f"Unsupported feature_standardization '{mode}'. "
        "Use 'none' or 'sequence'."
    )
