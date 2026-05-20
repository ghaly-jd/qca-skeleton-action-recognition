"""Dataset split helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.utils.io import write_json
from src.utils.paths import ensure_dir


def make_cross_subject_split(
    subjects: Iterable[int],
    sequence_ids: Iterable[str],
    *,
    train_subjects: Iterable[int],
    test_subjects: Iterable[int],
    dataset: str,
    seed: int | None = None,
) -> dict[str, object]:
    """Create a fixed cross-subject split, optionally shuffling index order."""
    subjects_array = np.asarray(list(subjects), dtype=int)
    sequence_id_list = list(sequence_ids)
    if len(subjects_array) != len(sequence_id_list):
        raise ValueError("subjects and sequence_ids must have the same length.")

    train_subject_set = {int(subject) for subject in train_subjects}
    test_subject_set = {int(subject) for subject in test_subjects}
    if train_subject_set & test_subject_set:
        raise ValueError("train_subjects and test_subjects must be disjoint.")

    train_indices = np.flatnonzero(np.isin(subjects_array, list(train_subject_set)))
    test_indices = np.flatnonzero(np.isin(subjects_array, list(test_subject_set)))

    if seed is not None:
        rng = np.random.default_rng(seed)
        train_indices = rng.permutation(train_indices)
        test_indices = rng.permutation(test_indices)

    return {
        "dataset": dataset,
        "protocol": "cross_subject",
        "seed": seed,
        "train_subjects": sorted(train_subject_set),
        "test_subjects": sorted(test_subject_set),
        "train_indices": train_indices.astype(int).tolist(),
        "test_indices": test_indices.astype(int).tolist(),
        "train_sequence_ids": [sequence_id_list[index] for index in train_indices],
        "test_sequence_ids": [sequence_id_list[index] for index in test_indices],
    }


def save_cross_subject_splits(
    output_dir: str | Path,
    subjects: Iterable[int],
    sequence_ids: Iterable[str],
    *,
    train_subjects: Iterable[int],
    test_subjects: Iterable[int],
    seeds: Iterable[int] = (),
    dataset: str,
) -> list[Path]:
    """Write the fixed split plus deterministic seeded order variants."""
    output_dir = ensure_dir(output_dir)
    subject_list = list(subjects)
    sequence_id_list = list(sequence_ids)

    paths = [
        write_json(
            output_dir / "msr_cross_subject.json",
            make_cross_subject_split(
                subject_list,
                sequence_id_list,
                train_subjects=train_subjects,
                test_subjects=test_subjects,
                dataset=dataset,
                seed=None,
            ),
        )
    ]

    for seed in seeds:
        paths.append(
            write_json(
                output_dir / f"msr_cross_subject_seed{int(seed)}.json",
                make_cross_subject_split(
                    subject_list,
                    sequence_id_list,
                    train_subjects=train_subjects,
                    test_subjects=test_subjects,
                    dataset=dataset,
                    seed=int(seed),
                ),
            )
        )

    return paths
