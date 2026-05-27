#!/usr/bin/env python
"""Run exact global projection-affinity 1-NN experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.data.validation import ProcessedDataset, load_processed_dataset
from src.distances.quantum_estimated_angles import (
    AFFINITY_NORMALIZATIONS,
    pairwise_exact_subspace_affinity_distances,
)
from src.eval.knn import predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import get_git_commit, utc_timestamp
from src.features.sequence_subspace import compute_subspaces, stack_bases
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


RESULT_FIELDNAMES = [
    "dataset",
    "seed",
    "method",
    "r",
    "affinity_normalization",
    "accuracy",
    "macro_f1",
    "runtime_sec",
    "train_size",
    "test_size",
    "skipped_sequences",
    "git_commit",
    "timestamp",
]

SUMMARY_FIELDNAMES = [
    "dataset",
    "method",
    "r",
    "affinity_normalization",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "runtime_mean",
]


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is currently implemented.")

    logger = get_logger("global_subspace_affinity")
    experiment_config = read_yaml(project_path(args.experiment_config))
    dataset_config = read_yaml(project_path(args.dataset_config))

    r_values = args.r_values or experiment_config["subspace"]["r_values"]
    seeds = args.seeds or experiment_config["seeds"]
    _validate_affinity_normalization(args.affinity_normalization)

    dataset = load_processed_dataset(_resolve_path(dataset_config["dataset"]["processed_dir"]))
    rows = run_experiment(
        dataset,
        seeds=[int(seed) for seed in seeds],
        r_values=[int(rank) for rank in r_values],
        min_frames_required=int(experiment_config["subspace"]["min_frames_required"]),
        center_sequence=bool(experiment_config["subspace"]["center_sequence"]),
        feature_standardization=args.feature_standardization,
        frame_l2_normalization=args.l2_normalize_frames,
        affinity_normalization=args.affinity_normalization,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        logger=logger,
    )

    output_path = _resolve_path(args.output)
    write_csv_rows(output_path, rows, fieldnames=RESULT_FIELDNAMES)
    logger.info("Wrote %s rows to %s.", len(rows), output_path)

    summary_rows = summarize_rows(rows)
    summary_path = _resolve_path(args.summary_output)
    write_csv_rows(summary_path, summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    logger.info("Wrote summary table to %s.", summary_path)


def run_experiment(
    dataset: ProcessedDataset,
    *,
    seeds: list[int],
    r_values: list[int],
    min_frames_required: int,
    center_sequence: bool,
    feature_standardization: str,
    frame_l2_normalization: bool,
    affinity_normalization: str,
    limit_train: int | None,
    limit_test: int | None,
    logger: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamp = utc_timestamp()
    git_commit = get_git_commit()

    for seed in seeds:
        split = _load_seed_split(seed)
        for rank in r_values:
            required_frames = max(min_frames_required, rank + 1)
            train_indices = _filter_indices_by_frame_count(
                split["train_indices"],
                dataset,
                required_frames=required_frames,
            )
            test_indices = _filter_indices_by_frame_count(
                split["test_indices"],
                dataset,
                required_frames=required_frames,
            )
            skipped = (
                len(split["train_indices"])
                + len(split["test_indices"])
                - len(train_indices)
                - len(test_indices)
            )
            if limit_train is not None:
                train_indices = train_indices[:limit_train]
            if limit_test is not None:
                test_indices = test_indices[:limit_test]
            if not train_indices or not test_indices:
                raise ValueError(
                    f"Empty train/test split after frame filtering for rank={rank}."
                )

            logger.info(
                "Running global projection affinity seed=%s r=%s train=%s test=%s.",
                seed,
                rank,
                len(train_indices),
                len(test_indices),
            )
            start = perf_counter()
            train_subspaces = _compute_indexed_subspaces(
                dataset,
                train_indices,
                rank=rank,
                center_sequence=center_sequence,
                min_frames_required=required_frames,
                feature_standardization=feature_standardization,
                frame_l2_normalization=frame_l2_normalization,
            )
            test_subspaces = _compute_indexed_subspaces(
                dataset,
                test_indices,
                rank=rank,
                center_sequence=center_sequence,
                min_frames_required=required_frames,
                feature_standardization=feature_standardization,
                frame_l2_normalization=frame_l2_normalization,
            )
            train_bases = stack_bases(train_subspaces)
            test_bases = stack_bases(test_subspaces)
            distance_matrix = pairwise_exact_subspace_affinity_distances(
                test_bases,
                train_bases,
                normalization=affinity_normalization,
            )
            y_train = dataset.labels[train_indices]
            y_test = dataset.labels[test_indices]
            result = predict_1nn_from_distances(distance_matrix, y_train)
            runtime = perf_counter() - start

            row = {
                "dataset": dataset.dataset,
                "seed": seed,
                "method": "global_projection_affinity",
                "r": rank,
                "affinity_normalization": affinity_normalization,
                "accuracy": round(accuracy(y_test, result.predictions), 6),
                "macro_f1": round(macro_f1(y_test, result.predictions), 6),
                "runtime_sec": round(runtime, 6),
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "skipped_sequences": skipped,
                "git_commit": git_commit,
                "timestamp": timestamp,
            }
            rows.append(row)
            logger.info(
                "seed=%s r=%s accuracy=%.4f macro_f1=%.4f runtime=%.2fs",
                seed,
                rank,
                row["accuracy"],
                row["macro_f1"],
                row["runtime_sec"],
            )

    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["method"]),
            int(row["r"]),
            str(row["affinity_normalization"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, method, rank, normalization), group_rows in sorted(groups.items()):
        accuracies = np.asarray([float(row["accuracy"]) for row in group_rows])
        macro_f1_values = np.asarray([float(row["macro_f1"]) for row in group_rows])
        runtimes = np.asarray([float(row["runtime_sec"]) for row in group_rows])
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "r": rank,
                "affinity_normalization": normalization,
                "accuracy_mean": round(float(accuracies.mean()), 6),
                "accuracy_std": round(float(accuracies.std(ddof=0)), 6),
                "macro_f1_mean": round(float(macro_f1_values.mean()), 6),
                "macro_f1_std": round(float(macro_f1_values.std(ddof=0)), 6),
                "runtime_mean": round(float(runtimes.mean()), 6),
            }
        )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msr_action3d"])
    parser.add_argument("--experiment-config", default="configs/experiment_main.yaml")
    parser.add_argument("--dataset-config", default="configs/msr_action3d.yaml")
    parser.add_argument("--r-values", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--affinity-normalization",
        default="projection_frobenius",
        choices=AFFINITY_NORMALIZATIONS,
    )
    parser.add_argument("--feature-standardization", default="none")
    parser.add_argument("--l2-normalize-frames", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument(
        "--output",
        default="results/raw/global_subspace_affinity.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/tables/global_subspace_affinity_summary.csv",
    )
    return parser.parse_args()


def _compute_indexed_subspaces(
    dataset: ProcessedDataset,
    indices: list[int],
    *,
    rank: int,
    center_sequence: bool,
    min_frames_required: int,
    feature_standardization: str,
    frame_l2_normalization: bool,
):
    return compute_subspaces(
        [dataset.sequences[index] for index in indices],
        [dataset.sequence_ids[index] for index in indices],
        rank=rank,
        center_sequence=center_sequence,
        min_frames_required=min_frames_required,
        feature_standardization=feature_standardization,
        frame_l2_normalization=frame_l2_normalization,
    )


def _filter_indices_by_frame_count(
    indices: list[int],
    dataset: ProcessedDataset,
    *,
    required_frames: int,
) -> list[int]:
    return [
        int(index)
        for index in indices
        if dataset.sequences[int(index)].shape[0] >= required_frames
    ]


def _load_seed_split(seed: int) -> dict[str, Any]:
    split_path = project_path("data", "splits", f"msr_cross_subject_seed{seed}.json")
    if not split_path.exists():
        split_path = project_path("data", "splits", "msr_cross_subject.json")
    return read_json(split_path)


def _validate_affinity_normalization(value: str) -> None:
    if value not in AFFINITY_NORMALIZATIONS:
        known = ", ".join(AFFINITY_NORMALIZATIONS)
        raise ValueError(f"Unknown affinity normalization '{value}'. Known values: {known}.")


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


if __name__ == "__main__":
    main()
