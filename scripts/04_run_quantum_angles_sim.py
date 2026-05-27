#!/usr/bin/env python
"""Run quantum-estimated subspace-affinity 1-NN experiments."""

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
    pairwise_quantum_subspace_affinity_distances,
)
from src.eval.knn import predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import get_git_commit, utc_timestamp
from src.features.motion_features import apply_feature_mode
from src.features.sequence_subspace import compute_subspaces, stack_bases
from src.quantum.overlap_estimation import SwapTestOverlapEstimator
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


RESULT_FIELDNAMES = [
    "dataset",
    "seed",
    "feature_mode",
    "r",
    "shots",
    "overlap_method",
    "simulator",
    "affinity_normalization",
    "subset",
    "train_per_class",
    "test_per_class",
    "accuracy",
    "macro_f1",
    "runtime_sec",
    "train_size",
    "test_size",
    "num_overlap_estimates",
    "num_cache_hits",
    "git_commit",
    "timestamp",
]

SUMMARY_FIELDNAMES = [
    "dataset",
    "r",
    "shots",
    "overlap_method",
    "simulator",
    "affinity_normalization",
    "subset",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "runtime_mean",
]


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is implemented for Phase 4.")

    logger = get_logger("quantum_angles")
    quantum_config = read_yaml(project_path(args.experiment_config))
    main_config = read_yaml(project_path(args.main_experiment_config))
    dataset_config = read_yaml(project_path(args.dataset_config))

    r_values = args.r_values or quantum_config["subspace_affinity"]["r_values_initial"]
    shots_values = args.shots or quantum_config["subspace_affinity"]["shots_initial"]
    seeds = args.seeds or main_config["seeds"]
    simulator = args.simulator or quantum_config["quantum"].get("simulator", "sampling")
    subset_enabled = _resolve_subset_flag(args.subset, quantum_config)
    train_per_class = args.train_per_class or quantum_config["subset"]["train_per_class"]
    test_per_class = args.test_per_class or quantum_config["subset"]["test_per_class"]

    feature_modes = args.feature_mode or ["position"]
    processed_dir = _resolve_path(dataset_config["dataset"]["processed_dir"])
    dataset = load_processed_dataset(processed_dir)
    rows = []
    for feature_mode in feature_modes:
        logger.info("Running feature_mode=%s", feature_mode)
        rows.extend(run_experiment(
            dataset,
            seeds=[int(seed) for seed in seeds],
            r_values=[int(rank) for rank in r_values],
            shots_values=[int(shots) for shots in shots_values],
            feature_mode=feature_mode,
            dataset_config=dataset_config,
            simulator=simulator,
            subset_enabled=subset_enabled,
            train_per_class=int(train_per_class),
            test_per_class=int(test_per_class),
            min_frames_required=int(main_config["subspace"]["min_frames_required"]),
            center_sequence=bool(main_config["subspace"]["center_sequence"]),
            affinity_normalization=args.affinity_normalization,
            limit_train=args.limit_train,
            limit_test=args.limit_test,
            cache_overlaps=not args.no_cache_overlaps,
            logger=logger,
        ))

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
    shots_values: list[int],
    feature_mode: str = "position",
    dataset_config: dict[str, Any] | None = None,
    simulator: str,
    subset_enabled: bool,
    train_per_class: int,
    test_per_class: int,
    min_frames_required: int,
    center_sequence: bool,
    affinity_normalization: str,
    limit_train: int | None,
    limit_test: int | None,
    cache_overlaps: bool,
    logger: Any,
) -> list[dict[str, Any]]:
    _validate_affinity_normalization(affinity_normalization)
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

            if subset_enabled:
                train_indices = _select_per_class(
                    train_indices,
                    dataset.labels,
                    train_per_class,
                )
                test_indices = _select_per_class(
                    test_indices,
                    dataset.labels,
                    test_per_class,
                )
            if limit_train is not None:
                train_indices = train_indices[:limit_train]
            if limit_test is not None:
                test_indices = test_indices[:limit_test]
            if not train_indices or not test_indices:
                raise ValueError("Train and test splits must both be non-empty.")

            extract_start = perf_counter()
            train_subspaces = _compute_indexed_subspaces(
                dataset,
                train_indices,
                rank=rank,
                center_sequence=center_sequence,
                min_frames_required=required_frames,
                feature_mode=feature_mode,
                dataset_config=dataset_config,
            )
            test_subspaces = _compute_indexed_subspaces(
                dataset,
                test_indices,
                rank=rank,
                center_sequence=center_sequence,
                min_frames_required=required_frames,
                feature_mode=feature_mode,
                dataset_config=dataset_config,
            )
            train_bases = stack_bases(train_subspaces)
            test_bases = stack_bases(test_subspaces)
            y_train = dataset.labels[train_indices]
            y_test = dataset.labels[test_indices]
            extraction_runtime = perf_counter() - extract_start

            for shots in shots_values:
                logger.info(
                    "Running quantum affinity seed=%s r=%s shots=%s simulator=%s train=%s test=%s.",
                    seed,
                    rank,
                    shots,
                    simulator,
                    len(train_indices),
                    len(test_indices),
                )
                estimator = SwapTestOverlapEstimator(
                    shots=shots,
                    simulator=simulator,
                    seed=_setting_seed(seed=seed, rank=rank, shots=shots),
                    cache=cache_overlaps,
                )
                classify_start = perf_counter()
                distance_matrix = pairwise_quantum_subspace_affinity_distances(
                    test_bases,
                    train_bases,
                    estimator=estimator,
                    normalization=affinity_normalization,
                )
                result = predict_1nn_from_distances(distance_matrix, y_train)
                runtime = extraction_runtime + (perf_counter() - classify_start)

                row = {
                    "dataset": dataset.dataset,
                    "seed": seed,
                    "feature_mode": feature_mode,
                    "r": rank,
                    "shots": shots,
                    "overlap_method": "swap_test",
                    "simulator": simulator,
                    "affinity_normalization": affinity_normalization,
                    "subset": subset_enabled,
                    "train_per_class": train_per_class if subset_enabled else "",
                    "test_per_class": test_per_class if subset_enabled else "",
                    "accuracy": round(accuracy(y_test, result.predictions), 6),
                    "macro_f1": round(macro_f1(y_test, result.predictions), 6),
                    "runtime_sec": round(runtime, 6),
                    "train_size": len(train_indices),
                    "test_size": len(test_indices),
                    "num_overlap_estimates": estimator.num_estimates,
                    "num_cache_hits": estimator.num_cache_hits,
                    "git_commit": git_commit,
                    "timestamp": timestamp,
                }
                rows.append(row)
                logger.info(
                    "seed=%s r=%s shots=%s accuracy=%.4f macro_f1=%.4f runtime=%.2fs",
                    seed,
                    rank,
                    shots,
                    row["accuracy"],
                    row["macro_f1"],
                    row["runtime_sec"],
                )

    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            int(row["r"]),
            int(row["shots"]),
            str(row["overlap_method"]),
            str(row["simulator"]),
            str(row["affinity_normalization"]),
            bool(row["subset"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (
        dataset,
        rank,
        shots,
        overlap_method,
        simulator,
        affinity_normalization,
        subset,
    ), group_rows in sorted(groups.items()):
        accuracies = np.asarray([float(row["accuracy"]) for row in group_rows])
        macro_f1_values = np.asarray([float(row["macro_f1"]) for row in group_rows])
        runtimes = np.asarray([float(row["runtime_sec"]) for row in group_rows])
        summary_rows.append(
            {
                "dataset": dataset,
                "r": rank,
                "shots": shots,
                "overlap_method": overlap_method,
                "simulator": simulator,
                "affinity_normalization": affinity_normalization,
                "subset": subset,
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
    parser.add_argument("--experiment-config", default="configs/experiment_quantum.yaml")
    parser.add_argument("--main-experiment-config", default="configs/experiment_main.yaml")
    parser.add_argument("--dataset-config", default="configs/msr_action3d.yaml")
    parser.add_argument("--r-values", nargs="+", type=int, default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--simulator",
        default=None,
        choices=["exact", "sampling", "qiskit_aer", "aer"],
        help="Use sampling for fast ideal shot-noise simulation or qiskit_aer for circuits.",
    )
    parser.add_argument(
        "--affinity-normalization",
        default="projection_frobenius",
        choices=AFFINITY_NORMALIZATIONS,
    )
    parser.add_argument(
        "--subset",
        nargs="?",
        const="true",
        default=None,
        help="Boolean flag. Accepts true/false; bare --subset means true.",
    )
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--no-cache-overlaps", action="store_true")
    parser.add_argument(
        "--feature-mode",
        nargs="+",
        default=None,
        metavar="MODE",
        help=(
            "Feature mode(s) to evaluate (default: position). "
            "One or more of: position velocity acceleration "
            "position_velocity bone_vectors bone_velocity."
        ),
    )
    parser.add_argument(
        "--output",
        default="results/raw/quantum_subspace_affinity.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/tables/quantum_subspace_affinity_summary.csv",
    )
    return parser.parse_args()


def _compute_indexed_subspaces(
    dataset: ProcessedDataset,
    indices: list[int],
    *,
    rank: int,
    center_sequence: bool,
    min_frames_required: int,
    feature_mode: str = "position",
    dataset_config: dict[str, Any] | None = None,
):
    sequences = [
        apply_feature_mode(dataset.sequences[index], feature_mode, dataset_config)
        for index in indices
    ]
    return compute_subspaces(
        sequences,
        [dataset.sequence_ids[index] for index in indices],
        rank=rank,
        center_sequence=center_sequence,
        min_frames_required=min_frames_required,
    )


def _select_per_class(
    indices: list[int],
    labels: np.ndarray,
    per_class: int,
) -> list[int]:
    if per_class <= 0:
        raise ValueError("per_class must be positive.")
    selected: list[int] = []
    values = labels.astype(int)
    for label in sorted({int(values[index]) for index in indices}):
        class_indices = [int(index) for index in indices if int(values[index]) == label]
        selected.extend(class_indices[:per_class])
    return selected


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


def _resolve_subset_flag(value: str | None, config: dict[str, Any]) -> bool:
    if value is None:
        return bool(config["subset"].get("enabled", True))
    return _parse_bool(value)


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value}")


def _setting_seed(*, seed: int, rank: int, shots: int) -> int:
    return int(seed * 1_000_003 + rank * 10_007 + shots)


def _validate_affinity_normalization(value: str) -> None:
    if value not in AFFINITY_NORMALIZATIONS:
        raise ValueError(
            f"Unknown affinity normalization '{value}'. "
            f"Known values: {', '.join(AFFINITY_NORMALIZATIONS)}."
        )


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


if __name__ == "__main__":
    main()
