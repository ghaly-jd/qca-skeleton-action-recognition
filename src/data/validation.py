"""Validation and summary helpers for processed datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import read_json, write_csv_rows


@dataclass(frozen=True)
class ProcessedDataset:
    """Loaded processed skeleton dataset."""

    dataset: str
    sequence_ids: list[str]
    sequences: list[np.ndarray]
    labels: np.ndarray
    subjects: np.ndarray
    metadata: dict[str, Any]


def load_processed_dataset(processed_dir: str | Path) -> ProcessedDataset:
    """Load processed arrays and metadata from a dataset directory."""
    processed_dir = Path(processed_dir)
    metadata = read_json(processed_dir / "metadata.json")
    sequence_ids = [item["sequence_id"] for item in metadata["sequences"]]

    with np.load(processed_dir / "sequences.npz") as archive:
        sequences = [np.array(archive[sequence_id]) for sequence_id in sequence_ids]

    labels = np.load(processed_dir / "labels.npy")
    subjects = np.load(processed_dir / "subjects.npy")
    return ProcessedDataset(
        dataset=str(metadata.get("dataset", processed_dir.name)),
        sequence_ids=sequence_ids,
        sequences=sequences,
        labels=labels,
        subjects=subjects,
        metadata=metadata,
    )


def validate_processed_dataset(
    processed_dir: str | Path,
    *,
    split_path: str | Path | None = None,
    expected_feature_dim: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a processed dataset and return a CSV-ready summary plus issues."""
    dataset = load_processed_dataset(processed_dir)
    issues: list[str] = []

    num_sequences = len(dataset.sequences)
    if len(dataset.labels) != num_sequences:
        issues.append("labels.npy length does not match number of sequences.")
    if len(dataset.subjects) != num_sequences:
        issues.append("subjects.npy length does not match number of sequences.")

    frame_counts: list[int] = []
    feature_dims: list[int] = []
    for sequence_id, sequence in zip(dataset.sequence_ids, dataset.sequences):
        if sequence.ndim != 2:
            issues.append(f"{sequence_id} is not a 2D T x D matrix.")
            continue
        frame_counts.append(int(sequence.shape[0]))
        feature_dims.append(int(sequence.shape[1]))
        if sequence.shape[0] <= 0:
            issues.append(f"{sequence_id} has no frames.")
        if not np.isfinite(sequence).all():
            issues.append(f"{sequence_id} contains non-finite values.")

    unique_feature_dims = sorted(set(feature_dims))
    if expected_feature_dim is not None and unique_feature_dims != [expected_feature_dim]:
        issues.append(
            f"Expected feature_dim {expected_feature_dim}, got {unique_feature_dims}."
        )

    if num_sequences == 0:
        issues.append("No processed sequences found.")

    summary = {
        "dataset": dataset.dataset,
        "num_sequences": num_sequences,
        "num_classes": int(len(set(dataset.labels.astype(int).tolist()))),
        "num_subjects": int(len(set(dataset.subjects.astype(int).tolist()))),
        "min_frames": int(min(frame_counts)) if frame_counts else 0,
        "max_frames": int(max(frame_counts)) if frame_counts else 0,
        "mean_frames": round(float(np.mean(frame_counts)), 3) if frame_counts else 0.0,
        "feature_dim": unique_feature_dims[0] if len(unique_feature_dims) == 1 else str(unique_feature_dims),
    }

    if split_path is not None:
        issues.extend(_validate_split(split_path, dataset))

    return summary, issues


def write_dataset_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    """Write the dataset summary table."""
    fieldnames = [
        "dataset",
        "num_sequences",
        "num_classes",
        "num_subjects",
        "min_frames",
        "max_frames",
        "mean_frames",
        "feature_dim",
    ]
    return write_csv_rows(path, [summary], fieldnames=fieldnames)


def _validate_split(path: str | Path, dataset: ProcessedDataset) -> list[str]:
    split = read_json(path)
    issues: list[str] = []
    num_sequences = len(dataset.sequences)

    train_indices = [int(index) for index in split.get("train_indices", [])]
    test_indices = [int(index) for index in split.get("test_indices", [])]
    train_set = set(train_indices)
    test_set = set(test_indices)

    if train_set & test_set:
        issues.append("Train/test split indices overlap.")
    if sorted(train_set | test_set) != list(range(num_sequences)):
        issues.append("Train/test split does not cover every sequence exactly once.")
    if any(index < 0 or index >= num_sequences for index in train_indices + test_indices):
        issues.append("Train/test split contains an out-of-range index.")

    train_subjects = {int(subject) for subject in split.get("train_subjects", [])}
    test_subjects = {int(subject) for subject in split.get("test_subjects", [])}
    if train_subjects & test_subjects:
        issues.append("Train/test subject lists overlap.")

    subjects = dataset.subjects.astype(int)
    if train_indices and not set(subjects[train_indices]).issubset(train_subjects):
        issues.append("Train indices contain subjects outside train_subjects.")
    if test_indices and not set(subjects[test_indices]).issubset(test_subjects):
        issues.append("Test indices contain subjects outside test_subjects.")

    return issues
