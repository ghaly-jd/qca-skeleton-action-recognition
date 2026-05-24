#!/usr/bin/env python
"""Compare exact and SWAP-test subspace affinity on the same subset."""

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
    pairwise_quantum_subspace_affinity_distances,
)
from src.eval.knn import KNNResult, predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import get_git_commit, utc_timestamp
from src.features.sequence_subspace import compute_subspaces, stack_bases
from src.quantum.overlap_estimation import SwapTestOverlapEstimator
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


RESULT_FIELDNAMES = [
    "dataset",
    "seed",
    "r",
    "shots",
    "overlap_method",
    "simulator",
    "affinity_normalization",
    "subset",
    "train_per_class",
    "test_per_class",
    "exact_accuracy",
    "exact_macro_f1",
    "quantum_accuracy",
    "quantum_macro_f1",
    "accuracy_gap_quantum_minus_exact",
    "macro_f1_gap_quantum_minus_exact",
    "mean_abs_distance_error",
    "rmse_distance_error",
    "max_abs_distance_error",
    "nearest_neighbor_agreement",
    "prediction_agreement",
    "exact_runtime_sec",
    "quantum_runtime_sec",
    "total_runtime_sec",
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
    "exact_accuracy_mean",
    "exact_accuracy_std",
    "quantum_accuracy_mean",
    "quantum_accuracy_std",
    "accuracy_gap_mean",
    "accuracy_gap_std",
    "exact_macro_f1_mean",
    "quantum_macro_f1_mean",
    "macro_f1_gap_mean",
    "mean_abs_distance_error_mean",
    "rmse_distance_error_mean",
    "max_abs_distance_error_mean",
    "nearest_neighbor_agreement_mean",
    "prediction_agreement_mean",
    "runtime_mean",
]


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is implemented for Phase 4.9.")

    logger = get_logger("quantum_exact_compare")
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

    dataset = load_processed_dataset(
        _resolve_path(dataset_config["dataset"]["processed_dir"])
    )
    rows = run_comparison(
        dataset,
        seeds=[int(seed) for seed in seeds],
        r_values=[int(rank) for rank in r_values],
        shots_values=[int(shots) for shots in shots_values],
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
    )

    output_path = _resolve_path(args.output)
    write_csv_rows(output_path, rows, fieldnames=RESULT_FIELDNAMES)
    logger.info("Wrote %s rows to %s.", len(rows), output_path)

    summary_rows = summarize_rows(rows)
    summary_path = _resolve_path(args.summary_output)
    write_csv_rows(summary_path, summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    logger.info("Wrote summary table to %s.", summary_path)


def run_comparison(
    dataset: ProcessedDataset,
    *,
    seeds: list[int],
    r_values: list[int],
    shots_values: list[int],
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
            train_indices, test_indices = _prepare_indices(
                dataset,
                split,
                rank=rank,
                min_frames_required=min_frames_required,
                subset_enabled=subset_enabled,
                train_per_class=train_per_class,
                test_per_class=test_per_class,
                limit_train=limit_train,
                limit_test=limit_test,
            )

            extract_start = perf_counter()
            train_subspaces = _compute_indexed_subspaces(
                dataset,
                train_indices,
                rank=rank,
                center_sequence=center_sequence,
                min_frames_required=max(min_frames_required, rank + 1),
            )
            test_subspaces = _compute_indexed_subspaces(
                dataset,
                test_indices,
                rank=rank,
                center_sequence=center_sequence,
                min_frames_required=max(min_frames_required, rank + 1),
            )
            train_bases = stack_bases(train_subspaces)
            test_bases = stack_bases(test_subspaces)
            y_train = dataset.labels[train_indices]
            y_test = dataset.labels[test_indices]
            extraction_runtime = perf_counter() - extract_start

            exact_start = perf_counter()
            exact_distances = pairwise_exact_subspace_affinity_distances(
                test_bases,
                train_bases,
                normalization=affinity_normalization,
            )
            exact_result = predict_1nn_from_distances(exact_distances, y_train)
            exact_runtime = extraction_runtime + (perf_counter() - exact_start)
            exact_accuracy = accuracy(y_test, exact_result.predictions)
            exact_f1 = macro_f1(y_test, exact_result.predictions)

            logger.info(
                "Exact affinity seed=%s r=%s accuracy=%.4f macro_f1=%.4f train=%s test=%s.",
                seed,
                rank,
                exact_accuracy,
                exact_f1,
                len(train_indices),
                len(test_indices),
            )

            for shots in shots_values:
                logger.info(
                    "Comparing quantum affinity seed=%s r=%s shots=%s simulator=%s.",
                    seed,
                    rank,
                    shots,
                    simulator,
                )
                estimator = SwapTestOverlapEstimator(
                    shots=shots,
                    simulator=simulator,
                    seed=_setting_seed(seed=seed, rank=rank, shots=shots),
                    cache=cache_overlaps,
                )
                quantum_start = perf_counter()
                quantum_distances = pairwise_quantum_subspace_affinity_distances(
                    test_bases,
                    train_bases,
                    estimator=estimator,
                    normalization=affinity_normalization,
                )
                quantum_result = predict_1nn_from_distances(quantum_distances, y_train)
                quantum_runtime = perf_counter() - quantum_start
                rows.append(
                    _comparison_row(
                        dataset=dataset.dataset,
                        seed=seed,
                        rank=rank,
                        shots=shots,
                        simulator=simulator,
                        affinity_normalization=affinity_normalization,
                        subset_enabled=subset_enabled,
                        train_per_class=train_per_class,
                        test_per_class=test_per_class,
                        y_test=y_test,
                        exact_result=exact_result,
                        quantum_result=quantum_result,
                        exact_distances=exact_distances,
                        quantum_distances=quantum_distances,
                        exact_runtime=exact_runtime,
                        quantum_runtime=quantum_runtime,
                        train_size=len(train_indices),
                        test_size=len(test_indices),
                        num_overlap_estimates=estimator.num_estimates,
                        num_cache_hits=estimator.num_cache_hits,
                        git_commit=git_commit,
                        timestamp=timestamp,
                    )
                )
                latest = rows[-1]
                logger.info(
                    "seed=%s r=%s shots=%s exact=%.4f quantum=%.4f mae=%.4f nn_agree=%.4f",
                    seed,
                    rank,
                    shots,
                    latest["exact_accuracy"],
                    latest["quantum_accuracy"],
                    latest["mean_abs_distance_error"],
                    latest["nearest_neighbor_agreement"],
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
        exact_accuracy = _values(group_rows, "exact_accuracy")
        quantum_accuracy = _values(group_rows, "quantum_accuracy")
        accuracy_gap = _values(group_rows, "accuracy_gap_quantum_minus_exact")
        exact_f1 = _values(group_rows, "exact_macro_f1")
        quantum_f1 = _values(group_rows, "quantum_macro_f1")
        macro_f1_gap = _values(group_rows, "macro_f1_gap_quantum_minus_exact")
        runtimes = _values(group_rows, "total_runtime_sec")
        summary_rows.append(
            {
                "dataset": dataset,
                "r": rank,
                "shots": shots,
                "overlap_method": overlap_method,
                "simulator": simulator,
                "affinity_normalization": affinity_normalization,
                "subset": subset,
                "exact_accuracy_mean": _mean(exact_accuracy),
                "exact_accuracy_std": _std(exact_accuracy),
                "quantum_accuracy_mean": _mean(quantum_accuracy),
                "quantum_accuracy_std": _std(quantum_accuracy),
                "accuracy_gap_mean": _mean(accuracy_gap),
                "accuracy_gap_std": _std(accuracy_gap),
                "exact_macro_f1_mean": _mean(exact_f1),
                "quantum_macro_f1_mean": _mean(quantum_f1),
                "macro_f1_gap_mean": _mean(macro_f1_gap),
                "mean_abs_distance_error_mean": _mean(
                    _values(group_rows, "mean_abs_distance_error")
                ),
                "rmse_distance_error_mean": _mean(
                    _values(group_rows, "rmse_distance_error")
                ),
                "max_abs_distance_error_mean": _mean(
                    _values(group_rows, "max_abs_distance_error")
                ),
                "nearest_neighbor_agreement_mean": _mean(
                    _values(group_rows, "nearest_neighbor_agreement")
                ),
                "prediction_agreement_mean": _mean(
                    _values(group_rows, "prediction_agreement")
                ),
                "runtime_mean": _mean(runtimes),
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
        "--output",
        default="results/raw/quantum_exact_subset_comparison.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/tables/quantum_exact_subset_comparison_summary.csv",
    )
    return parser.parse_args()


def _comparison_row(
    *,
    dataset: str,
    seed: int,
    rank: int,
    shots: int,
    simulator: str,
    affinity_normalization: str,
    subset_enabled: bool,
    train_per_class: int,
    test_per_class: int,
    y_test: np.ndarray,
    exact_result: KNNResult,
    quantum_result: KNNResult,
    exact_distances: np.ndarray,
    quantum_distances: np.ndarray,
    exact_runtime: float,
    quantum_runtime: float,
    train_size: int,
    test_size: int,
    num_overlap_estimates: int,
    num_cache_hits: int,
    git_commit: str,
    timestamp: str,
) -> dict[str, Any]:
    exact_accuracy = accuracy(y_test, exact_result.predictions)
    quantum_accuracy = accuracy(y_test, quantum_result.predictions)
    exact_f1 = macro_f1(y_test, exact_result.predictions)
    quantum_f1 = macro_f1(y_test, quantum_result.predictions)
    distance_error = quantum_distances - exact_distances
    return {
        "dataset": dataset,
        "seed": seed,
        "r": rank,
        "shots": shots,
        "overlap_method": "swap_test",
        "simulator": simulator,
        "affinity_normalization": affinity_normalization,
        "subset": subset_enabled,
        "train_per_class": train_per_class if subset_enabled else "",
        "test_per_class": test_per_class if subset_enabled else "",
        "exact_accuracy": round(exact_accuracy, 6),
        "exact_macro_f1": round(exact_f1, 6),
        "quantum_accuracy": round(quantum_accuracy, 6),
        "quantum_macro_f1": round(quantum_f1, 6),
        "accuracy_gap_quantum_minus_exact": round(quantum_accuracy - exact_accuracy, 6),
        "macro_f1_gap_quantum_minus_exact": round(quantum_f1 - exact_f1, 6),
        "mean_abs_distance_error": round(float(np.mean(np.abs(distance_error))), 6),
        "rmse_distance_error": round(
            float(np.sqrt(np.mean(np.square(distance_error)))),
            6,
        ),
        "max_abs_distance_error": round(float(np.max(np.abs(distance_error))), 6),
        "nearest_neighbor_agreement": round(
            float(np.mean(exact_result.nearest_indices == quantum_result.nearest_indices)),
            6,
        ),
        "prediction_agreement": round(
            float(np.mean(exact_result.predictions == quantum_result.predictions)),
            6,
        ),
        "exact_runtime_sec": round(exact_runtime, 6),
        "quantum_runtime_sec": round(quantum_runtime, 6),
        "total_runtime_sec": round(exact_runtime + quantum_runtime, 6),
        "train_size": train_size,
        "test_size": test_size,
        "num_overlap_estimates": num_overlap_estimates,
        "num_cache_hits": num_cache_hits,
        "git_commit": git_commit,
        "timestamp": timestamp,
    }


def _prepare_indices(
    dataset: ProcessedDataset,
    split: dict[str, Any],
    *,
    rank: int,
    min_frames_required: int,
    subset_enabled: bool,
    train_per_class: int,
    test_per_class: int,
    limit_train: int | None,
    limit_test: int | None,
) -> tuple[list[int], list[int]]:
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
        train_indices = _select_per_class(train_indices, dataset.labels, train_per_class)
        test_indices = _select_per_class(test_indices, dataset.labels, test_per_class)
    if limit_train is not None:
        train_indices = train_indices[:limit_train]
    if limit_test is not None:
        test_indices = test_indices[:limit_test]
    if not train_indices or not test_indices:
        raise ValueError("Train and test splits must both be non-empty.")
    return train_indices, test_indices


def _compute_indexed_subspaces(
    dataset: ProcessedDataset,
    indices: list[int],
    *,
    rank: int,
    center_sequence: bool,
    min_frames_required: int,
):
    return compute_subspaces(
        [dataset.sequences[index] for index in indices],
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
        selected.extend([int(index) for index in indices if int(values[index]) == label][:per_class])
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


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _mean(values: np.ndarray) -> float:
    return round(float(np.mean(values)), 6)


def _std(values: np.ndarray) -> float:
    return round(float(np.std(values, ddof=0)), 6)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


if __name__ == "__main__":
    main()
