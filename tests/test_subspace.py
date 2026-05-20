from __future__ import annotations

import numpy as np
import pytest

from src.features.sequence_subspace import compute_sequence_subspace, is_orthonormal


def test_sequence_subspace_basis_is_d_by_r_and_orthonormal():
    rng = np.random.default_rng(0)
    sequence = rng.normal(size=(20, 6))

    subspace = compute_sequence_subspace(sequence, rank=3, sequence_id="toy")

    assert subspace.basis.shape == (6, 3)
    assert subspace.singular_values.shape == (3,)
    assert is_orthonormal(subspace.basis)


def test_sequence_subspace_rejects_too_few_frames_for_centered_rank():
    sequence = np.ones((3, 6))

    with pytest.raises(ValueError, match="need at least rank"):
        compute_sequence_subspace(sequence, rank=3, min_frames_required=1)


def test_sign_flipped_input_spans_same_rank_one_subspace():
    t = np.linspace(-1.0, 1.0, 20)
    sequence = np.column_stack([t, 2 * t, -t])

    subspace_a = compute_sequence_subspace(sequence, rank=1)
    subspace_b = compute_sequence_subspace(-sequence, rank=1)

    overlap = abs(float(subspace_a.basis[:, 0] @ subspace_b.basis[:, 0]))
    assert np.isclose(overlap, 1.0)
