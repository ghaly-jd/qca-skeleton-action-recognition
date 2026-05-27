"""Motion feature transforms for skeleton sequences.

Each function takes a sequence ``X`` of shape ``(T, D)`` where ``T`` is the
number of frames and ``D`` is the feature dimension (60 for MSR Action3D: 20
joints × 3 coordinates).

Available transforms
--------------------
``position(X)``
    Identity; returns raw joint positions.  Output shape: ``(T, D)``.

``velocity(X)``
    First-order temporal finite differences.  The first frame is padded with
    zeros so the output has the same length as the input.  Output: ``(T, D)``.

``acceleration(X)``
    Second-order temporal finite differences.  The first two frames are padded
    with zeros.  Output: ``(T, D)``.

``position_velocity(X)``
    Concatenation of position and velocity along the feature axis.
    Output: ``(T, 2*D)``.

``bone_vectors(X, joint_hierarchy)``
    Per-bone offset vectors computed as ``child_joint - parent_joint`` for each
    edge in *joint_hierarchy*.  Output shape: ``(T, n_bones * 3)`` where
    ``n_bones = len(joint_hierarchy)`` and each joint has 3 coordinates.
    **This function requires the joint positions to be stored in joint-major
    order**: the first three columns are joint-0's x/y/z, the next three are
    joint-1's x/y/z, etc.

``bone_velocity(X, joint_hierarchy)``
    Concatenation of ``bone_vectors`` and their velocity.
    Output: ``(T, 2 * n_bones * 3)``.

Entry point
-----------
``apply_feature_mode(X, mode, config)``
    Dispatcher that calls the right function based on the string *mode*.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------


def position(X: np.ndarray) -> np.ndarray:
    """Return *X* unchanged (identity transform).

    Parameters
    ----------
    X:
        Input sequence of shape ``(T, D)``.

    Returns
    -------
    np.ndarray
        Same array, shape ``(T, D)``.
    """
    return np.asarray(X, dtype=float)


def velocity(X: np.ndarray) -> np.ndarray:
    """First-order temporal finite differences with zero-padding at frame 0.

    Parameters
    ----------
    X:
        Input sequence of shape ``(T, D)``.

    Returns
    -------
    np.ndarray
        Velocity sequence of shape ``(T, D)``.  ``result[0] = 0``,
        ``result[t] = X[t] - X[t-1]`` for ``t >= 1``.
    """
    X = np.asarray(X, dtype=float)
    T, D = X.shape
    out = np.zeros_like(X)
    if T > 1:
        out[1:] = X[1:] - X[:-1]
    return out


def acceleration(X: np.ndarray) -> np.ndarray:
    """Second-order temporal finite differences with zero-padding at frames 0–1.

    Computes ``X[t] - 2*X[t-1] + X[t-2]`` directly (not as repeated velocity)
    so that the zero-padding at frames 0 and 1 is clean.

    Parameters
    ----------
    X:
        Input sequence of shape ``(T, D)``.

    Returns
    -------
    np.ndarray
        Acceleration sequence of shape ``(T, D)``.  ``result[0:2] = 0``,
        ``result[t] = X[t] - 2*X[t-1] + X[t-2]`` for ``t >= 2``.
    """
    X = np.asarray(X, dtype=float)
    T, D = X.shape
    out = np.zeros_like(X)
    if T > 2:
        out[2:] = X[2:] - 2.0 * X[1:-1] + X[:-2]
    return out


def position_velocity(X: np.ndarray) -> np.ndarray:
    """Concatenate joint positions and velocity along the feature axis.

    Parameters
    ----------
    X:
        Input sequence of shape ``(T, D)``.

    Returns
    -------
    np.ndarray
        Shape ``(T, 2*D)``.
    """
    X = np.asarray(X, dtype=float)
    return np.concatenate([position(X), velocity(X)], axis=-1)


def bone_vectors(
    X: np.ndarray,
    joint_hierarchy: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Compute per-bone offset vectors (child_joint − parent_joint).

    Joint positions are assumed to be stored in **joint-major order**: the first
    three columns are joint-0's (x, y, z), the next three are joint-1's, etc.
    So joint *j*'s coordinates occupy columns ``[3*j : 3*j+3]``.

    Parameters
    ----------
    X:
        Sequence of shape ``(T, n_joints * 3)``.
    joint_hierarchy:
        A list of ``(parent_index, child_index)`` pairs defining the skeleton
        tree.  Indices are 0-based joint indices.

    Returns
    -------
    np.ndarray
        Shape ``(T, n_bones * 3)`` where ``n_bones = len(joint_hierarchy)``.
        Bones are ordered as in *joint_hierarchy*.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    n_bones = len(joint_hierarchy)
    out = np.empty((T, n_bones * 3), dtype=float)
    for b_idx, (parent, child) in enumerate(joint_hierarchy):
        parent_cols = slice(3 * parent, 3 * parent + 3)
        child_cols = slice(3 * child, 3 * child + 3)
        out[:, 3 * b_idx : 3 * b_idx + 3] = X[:, child_cols] - X[:, parent_cols]
    return out


def bone_velocity(
    X: np.ndarray,
    joint_hierarchy: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Concatenate bone vectors and their velocity.

    Parameters
    ----------
    X:
        Sequence of shape ``(T, n_joints * 3)``.
    joint_hierarchy:
        Same as in :func:`bone_vectors`.

    Returns
    -------
    np.ndarray
        Shape ``(T, 2 * n_bones * 3)``.
    """
    bv = bone_vectors(X, joint_hierarchy)
    return np.concatenate([bv, velocity(bv)], axis=-1)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def apply_feature_mode(
    X: np.ndarray,
    mode: str,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply the named feature transform to sequence *X*.

    Parameters
    ----------
    X:
        Sequence of shape ``(T, D)``.
    mode:
        One of ``"position"``, ``"velocity"``, ``"acceleration"``,
        ``"position_velocity"``, ``"bone_vectors"``, ``"bone_velocity"``.
    config:
        Dataset config dict (optional).  Required for modes that need the bone
        hierarchy (``"bone_vectors"`` and ``"bone_velocity"``).  The config
        must have a ``"bone_hierarchy"`` key containing a list of
        ``[parent_idx, child_idx]`` pairs.

    Returns
    -------
    np.ndarray
        Transformed sequence.

    Raises
    ------
    ValueError
        If *mode* is unknown or *config* is missing for bone-vector modes.
    """
    mode = mode.strip().lower()

    if mode == "position":
        return position(X)
    if mode == "velocity":
        return velocity(X)
    if mode == "acceleration":
        return acceleration(X)
    if mode == "position_velocity":
        return position_velocity(X)
    if mode in ("bone_vectors", "bone_velocity"):
        if config is None or "bone_hierarchy" not in config:
            raise ValueError(
                f"Feature mode '{mode}' requires a config dict with a "
                "'bone_hierarchy' key.  Pass the dataset YAML config."
            )
        hierarchy = [tuple(pair) for pair in config["bone_hierarchy"]]
        if mode == "bone_vectors":
            return bone_vectors(X, hierarchy)
        return bone_velocity(X, hierarchy)

    raise ValueError(
        f"Unknown feature mode '{mode}'.  Valid modes: "
        "position, velocity, acceleration, position_velocity, "
        "bone_vectors, bone_velocity."
    )
