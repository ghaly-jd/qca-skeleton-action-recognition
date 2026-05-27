"""Tests for src/features/motion_features.py."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.features.motion_features import (
    acceleration,
    apply_feature_mode,
    bone_velocity,
    bone_vectors,
    position,
    position_velocity,
    velocity,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# MSR Action3D standard bone hierarchy (19 bones, 20 joints).
MSR_HIERARCHY = [
    (0, 1), (1, 2), (2, 3),
    (2, 4), (4, 5), (5, 6), (6, 7),
    (2, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

T, D = 20, 60  # 20 frames, 20 joints × 3 coords


@pytest.fixture
def random_seq():
    rng = np.random.default_rng(42)
    return rng.standard_normal((T, D))


@pytest.fixture
def constant_seq():
    return np.ones((T, D)) * 0.5


@pytest.fixture
def linear_seq():
    """Linearly increasing sequence: X[t] = t * step."""
    step = 0.1
    return np.arange(T)[:, None] * step * np.ones((1, D))


@pytest.fixture
def msr_config():
    with open("configs/msr_action3d.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def test_position_shape(self, random_seq):
        assert position(random_seq).shape == (T, D)

    def test_velocity_shape(self, random_seq):
        assert velocity(random_seq).shape == (T, D)

    def test_acceleration_shape(self, random_seq):
        assert acceleration(random_seq).shape == (T, D)

    def test_position_velocity_shape(self, random_seq):
        assert position_velocity(random_seq).shape == (T, 2 * D)

    def test_bone_vectors_shape(self, random_seq):
        out = bone_vectors(random_seq, MSR_HIERARCHY)
        assert out.shape == (T, 19 * 3)  # 57

    def test_bone_velocity_shape(self, random_seq):
        out = bone_velocity(random_seq, MSR_HIERARCHY)
        assert out.shape == (T, 2 * 19 * 3)  # 114


# ---------------------------------------------------------------------------
# velocity tests
# ---------------------------------------------------------------------------


class TestVelocity:
    def test_first_frame_is_zero(self, random_seq):
        """velocity[0] must be zero (zero-padding)."""
        vel = velocity(random_seq)
        np.testing.assert_array_equal(vel[0], np.zeros(D))

    def test_constant_sequence_all_zero(self, constant_seq):
        """Velocity of a constant sequence is all zeros."""
        vel = velocity(constant_seq)
        np.testing.assert_allclose(vel, 0.0, atol=1e-12)

    def test_linear_sequence_constant_nonzero(self, linear_seq):
        """Velocity of a linear sequence is constant for frames >= 1."""
        vel = velocity(linear_seq)
        # All rows from index 1 onward should be identical (same constant step).
        expected = np.tile(vel[1], (T - 1, 1))  # shape (T-1, D) — explicit broadcast
        np.testing.assert_allclose(vel[1:], expected, atol=1e-12)
        # The value should be non-zero.
        assert np.all(np.abs(vel[1]) > 0)

    def test_output_values_correct(self, random_seq):
        """Explicit check: vel[t] == X[t] - X[t-1] for t >= 1."""
        vel = velocity(random_seq)
        np.testing.assert_allclose(vel[1:], random_seq[1:] - random_seq[:-1], atol=1e-12)


# ---------------------------------------------------------------------------
# acceleration tests
# ---------------------------------------------------------------------------


class TestAcceleration:
    def test_first_two_frames_zero(self, random_seq):
        """Acceleration at frames 0 and 1 must be zero (zero-padding)."""
        acc = acceleration(random_seq)
        np.testing.assert_array_equal(acc[0], np.zeros(D))
        np.testing.assert_array_equal(acc[1], np.zeros(D))

    def test_linear_sequence_all_zero(self, linear_seq):
        """Acceleration of a linear sequence is zero (constant velocity)."""
        acc = acceleration(linear_seq)
        np.testing.assert_allclose(acc, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# bone_vectors tests
# ---------------------------------------------------------------------------


class TestBoneVectors:
    def test_static_skeleton_same_at_every_frame(self):
        """A skeleton that doesn't move has the same bone vectors everywhere."""
        X = np.tile(np.arange(D, dtype=float), (T, 1))  # same frame repeated
        bv = bone_vectors(X, MSR_HIERARCHY)
        # Tile first-frame row to compare shapes explicitly (no broadcasting).
        expected = np.tile(bv[0], (T, 1))
        np.testing.assert_allclose(bv, expected, atol=1e-12)

    def test_invariant_to_global_translation(self, random_seq, msr_config):
        """bone_vectors(X + global_3d_offset) == bone_vectors(X).

        A *global* 3D translation applies the same (dx, dy, dz) shift to every
        joint.  Because bone vectors are child − parent, this cancels out.
        Note: adding a *per-joint* arbitrary offset (shape D) does NOT cancel,
        so we must use a single 3D offset tiled across all joints.
        """
        rng = np.random.default_rng(7)
        offset_3d = rng.standard_normal(3)                   # shape (3,)
        offset_full = np.tile(offset_3d, D // 3)             # shape (D,)
        hierarchy = [tuple(p) for p in msr_config["bone_hierarchy"]]
        bv_orig = bone_vectors(random_seq, hierarchy)
        bv_shifted = bone_vectors(random_seq + offset_full, hierarchy)
        np.testing.assert_allclose(bv_orig, bv_shifted, atol=1e-10)

    def test_correct_bone_count(self, random_seq):
        bv = bone_vectors(random_seq, MSR_HIERARCHY)
        assert bv.shape[1] == len(MSR_HIERARCHY) * 3

    def test_bone_vector_values(self):
        """Explicit check: bone[b] == joint[child] - joint[parent]."""
        X = np.random.randn(5, D)
        bv = bone_vectors(X, MSR_HIERARCHY)
        for b_idx, (parent, child) in enumerate(MSR_HIERARCHY):
            expected = X[:, 3 * child: 3 * child + 3] - X[:, 3 * parent: 3 * parent + 3]
            np.testing.assert_allclose(bv[:, 3 * b_idx: 3 * b_idx + 3], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# apply_feature_mode dispatcher tests
# ---------------------------------------------------------------------------


class TestApplyFeatureMode:
    @pytest.mark.parametrize("mode, expected_cols", [
        ("position", D),
        ("velocity", D),
        ("acceleration", D),
        ("position_velocity", 2 * D),
        ("bone_vectors", 57),
        ("bone_velocity", 114),
    ])
    def test_shape_via_dispatcher(self, random_seq, msr_config, mode, expected_cols):
        out = apply_feature_mode(random_seq, mode, msr_config)
        assert out.shape == (T, expected_cols)

    def test_unknown_mode_raises(self, random_seq):
        with pytest.raises(ValueError, match="Unknown feature mode"):
            apply_feature_mode(random_seq, "turbojet")

    def test_bone_modes_without_config_raise(self, random_seq):
        with pytest.raises(ValueError, match="bone_hierarchy"):
            apply_feature_mode(random_seq, "bone_vectors", config=None)
        with pytest.raises(ValueError, match="bone_hierarchy"):
            apply_feature_mode(random_seq, "bone_velocity", config=None)

    def test_position_returns_same_values(self, random_seq, msr_config):
        """Position mode must be a pure identity transform."""
        out = apply_feature_mode(random_seq, "position", msr_config)
        np.testing.assert_array_equal(out, random_seq)
