#!/usr/bin/env python
"""Prepare raw skeleton datasets into reproducible processed arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.data.msr_loader import load_msr_action3d
from src.data.preprocessing import PreprocessingConfig, preprocess_skeleton_sequence
from src.data.splits import save_cross_subject_splits
from src.eval.result_writer import utc_timestamp
from src.utils.io import read_yaml, write_json
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir, project_path


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is implemented in Phase 1.")

    logger = get_logger("prepare_dataset")
    config = read_yaml(project_path(args.config))
    dataset_config = config["dataset"]
    split_config = config["split"]
    preprocessing_config = PreprocessingConfig.from_mapping(config.get("preprocessing"))

    raw_dir = _resolve_path(args.raw_dir or dataset_config["raw_dir"])
    processed_dir = ensure_dir(_resolve_path(args.processed_dir or dataset_config["processed_dir"]))

    raw_sequences, skeleton_source = load_msr_action3d(
        raw_dir,
        num_joints=int(dataset_config["num_joints"]),
        prefer_real3d=not args.use_depth_coordinates,
    )

    sequences_by_id: dict[str, np.ndarray] = {}
    sequence_ids: list[str] = []
    labels: list[int] = []
    subjects: list[int] = []
    repetition_ids: list[int] = []
    sequence_metadata: list[dict[str, object]] = []
    skipped_sequences: list[dict[str, object]] = []

    for raw_sequence in raw_sequences:
        try:
            matrix, preprocess_info = preprocess_skeleton_sequence(
                raw_sequence.joints,
                raw_sequence.confidence,
                config=preprocessing_config,
            )
        except ValueError as exc:
            if "No valid skeleton frames remain" not in str(exc):
                raise
            skipped_sequences.append(
                {
                    "sequence_id": raw_sequence.sequence_id,
                    "subject_id": raw_sequence.subject_id,
                    "action_id": raw_sequence.action_id,
                    "repetition_id": raw_sequence.repetition_id,
                    "source_file": str(raw_sequence.path.relative_to(raw_dir)),
                    "reason": str(exc),
                }
            )
            logger.warning("Skipping %s: %s", raw_sequence.sequence_id, exc)
            continue

        sequences_by_id[raw_sequence.sequence_id] = matrix
        sequence_ids.append(raw_sequence.sequence_id)
        labels.append(raw_sequence.action_id)
        subjects.append(raw_sequence.subject_id)
        repetition_ids.append(raw_sequence.repetition_id)
        sequence_metadata.append(
            {
                "sequence_id": raw_sequence.sequence_id,
                "subject_id": raw_sequence.subject_id,
                "action_id": raw_sequence.action_id,
                "repetition_id": raw_sequence.repetition_id,
                "num_frames": preprocess_info["num_frames"],
                "original_frames": preprocess_info["original_frames"],
                "removed_frames": preprocess_info["removed_frames"],
                "feature_dim": preprocess_info["feature_dim"],
                "source_file": str(raw_sequence.path.relative_to(raw_dir)),
                "root_joint_index_zero_based": preprocess_info["root_joint_index_zero_based"],
                "scale_factor": preprocess_info["scale_factor"],
            }
        )

    np.savez_compressed(processed_dir / "sequences.npz", **sequences_by_id)
    np.save(processed_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(processed_dir / "subjects.npy", np.asarray(subjects, dtype=np.int64))
    np.save(processed_dir / "repetition_ids.npy", np.asarray(repetition_ids, dtype=np.int64))
    np.save(processed_dir / "sequence_ids.npy", np.asarray(sequence_ids))

    metadata = {
        "dataset": args.dataset,
        "generated_at": utc_timestamp(),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "skeleton_source": skeleton_source,
        "coordinate_source": "real3d" if not args.use_depth_coordinates else "depth_pixel",
        "num_sequences": len(sequence_ids),
        "num_joints": int(dataset_config["num_joints"]),
        "coordinate_dim": int(dataset_config["coordinate_dim"]),
        "feature_dim": int(dataset_config["feature_dim"]),
        "flatten_order": dataset_config["flatten_order"],
        "preprocessing": preprocessing_config.to_dict(),
        "sequences": sequence_metadata,
        "skipped_sequences": skipped_sequences,
    }
    write_json(processed_dir / "metadata.json", metadata)

    split_paths = save_cross_subject_splits(
        project_path("data", "splits"),
        subjects,
        sequence_ids,
        train_subjects=split_config["train_subjects"],
        test_subjects=split_config["test_subjects"],
        seeds=split_config.get("seeds", []),
        dataset=args.dataset,
    )

    logger.info("Prepared %s sequences from %s.", len(sequence_ids), raw_dir)
    if skipped_sequences:
        logger.info("Skipped %s invalid sequences.", len(skipped_sequences))
    logger.info("Wrote processed arrays to %s.", processed_dir)
    logger.info("Wrote %s split files under %s.", len(split_paths), split_paths[0].parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msr_action3d"])
    parser.add_argument("--config", default="configs/msr_action3d.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument(
        "--use-depth-coordinates",
        action="store_true",
        help="Use MSRAction3DSkeleton(20joints) depth/pixel skeletons instead of Real3D.",
    )
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


if __name__ == "__main__":
    main()
