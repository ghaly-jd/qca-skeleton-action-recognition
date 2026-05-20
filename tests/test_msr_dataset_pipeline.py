from __future__ import annotations

import numpy as np

from src.data.msr_loader import load_skeleton_file, parse_msr_filename
from src.data.preprocessing import PreprocessingConfig, preprocess_skeleton_sequence
from src.data.splits import make_cross_subject_split


def test_parse_msr_filename_accepts_real3d_and_depth_names():
    real3d = parse_msr_filename("a01_s02_e03_skeleton3D.txt")
    depth = parse_msr_filename("a04_s05_e01_skeleton.txt")

    assert real3d["sequence_id"] == "a01_s02_e03"
    assert real3d["action_id"] == 1
    assert real3d["subject_id"] == 2
    assert real3d["repetition_id"] == 3
    assert real3d["is_real3d"] is True
    assert depth["is_real3d"] is False


def test_load_skeleton_file_returns_frames_joints_and_confidence(tmp_path):
    rows = np.arange(2 * 20 * 4, dtype=float).reshape(40, 4)
    path = tmp_path / "a01_s01_e01_skeleton3D.txt"
    np.savetxt(path, rows)

    sequence = load_skeleton_file(path)

    assert sequence.sequence_id == "a01_s01_e01"
    assert sequence.joints.shape == (2, 20, 3)
    assert sequence.confidence is not None
    assert sequence.confidence.shape == (2, 20)


def test_preprocess_skeleton_sequence_flattens_to_60_features():
    joints = np.ones((3, 20, 3), dtype=float)
    joints[:, :, 0] = np.arange(20)
    confidence = np.ones((3, 20), dtype=float)

    matrix, info = preprocess_skeleton_sequence(
        joints,
        confidence,
        config=PreprocessingConfig(scale_normalize=False),
    )

    assert matrix.shape == (3, 60)
    assert info["feature_dim"] == 60
    assert info["removed_frames"] == 0


def test_cross_subject_split_is_disjoint_and_complete():
    split = make_cross_subject_split(
        subjects=[1, 2, 3, 4],
        sequence_ids=["a", "b", "c", "d"],
        train_subjects=[1, 3],
        test_subjects=[2, 4],
        dataset="msr_action3d",
    )

    assert split["train_indices"] == [0, 2]
    assert split["test_indices"] == [1, 3]
    assert set(split["train_indices"]).isdisjoint(split["test_indices"])
