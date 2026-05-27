#!/usr/bin/env python
"""Run exact-vs-SWAP Local Subspace-DTW subset experiments."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.data.validation import ProcessedDataset, load_processed_dataset
from src.distances.dtw import window_ratio_label
from src.distances.local_sdtw import pairwise_local_sdtw_distances
from src.distances.quantum_estimated_angles import AFFINITY_NORMALIZATIONS
from src.distances.quantum_local_sdtw import (
    QuantumLocalSdtwComparison,
    pairwise_quantum_exact_local_sdtw_comparison,
)
from src.eval.knn import KNNResult, predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import get_git_commit, utc_timestamp
from src.features.local_subspace import FEATURE_MODES, compute_local_subspace_sequence
from src.quantum.overlap_estimation import SwapTestOverlapEstimator
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


RESULT_FIELDNAMES = [
    "dataset",
    "seed",
    "method",
    "feature_mode",
    "window_length",
    "stride",
    "r",
    "shots",
    "overlap_method",
    "simulator",
    "affinity_normalization",
    "subset",
    "train_per_class",
    "test_per_class",
    "dtw_window",
    "normalize_by_path_length",
    "exact_accuracy",
    "exact_macro_f1",
    "quantum_accuracy",
    "quantum_macro_f1",
    "accuracy_gap_quantum_minus_exact",
    "macro_f1_gap_quantum_minus_exact",
    "mean_abs_distance_error",
    "rmse_distance_error",
    "max_abs_distance_error",
    "mean_cost_matrix_mae",
    "mean_cost_matrix_rmse",
    "max_cost_matrix_mae",
    "nearest_neighbor_agreement",
    "prediction_agreement",
    "feature_runtime_sec",
    "exact_runtime_sec",
    "quantum_runtime_sec",
    "total_runtime_sec",
    "train_size",
    "test_size",
    "skipped_sequences",
    "num_train_windows_mean",
    "num_test_windows_mean",
    "num_dtw_pairs",
    "num_local_costs",
    "num_overlap_estimates",
    "num_cache_hits",
    "git_commit",
    "timestamp",
]

SUMMARY_FIELDNAMES = [
    "dataset",
    "method",
    "feature_mode",
    "window_length",
    "stride",
    "r",
    "shots",
    "overlap_method",
    "simulator",
    "affinity_normalization",
    "subset",
    "train_per_class",
    "test_per_class",
    "dtw_window",
    "normalize_by_path_length",
    "exact_accuracy_mean",
    "exact_accuracy_std",
    "quantum_accuracy_mean",
    "quantum_accuracy_std",
    "accuracy_gap_mean",
    "accuracy_gap_std",
    "exact_macro_f1_mean",
    "exact_macro_f1_std",
    "quantum_macro_f1_mean",
    "quantum_macro_f1_std",
    "macro_f1_gap_mean",
    "macro_f1_gap_std",
    "mean_abs_distance_error_mean",
    "mean_abs_distance_error_std",
    "rmse_distance_error_mean",
    "rmse_distance_error_std",
    "max_abs_distance_error_mean",
    "mean_cost_matrix_mae_mean",
    "mean_cost_matrix_rmse_mean",
    "max_cost_matrix_mae_mean",
    "nearest_neighbor_agreement_mean",
    "nearest_neighbor_agreement_std",
    "prediction_agreement_mean",
    "prediction_agreement_std",
    "feature_runtime_mean",
    "exact_runtime_mean",
    "quantum_runtime_mean",
    "total_runtime_mean",
    "train_size_mean",
    "test_size_mean",
    "num_train_windows_mean",
    "num_test_windows_mean",
    "num_dtw_pairs_mean",
    "num_local_costs_mean",
    "num_overlap_estimates_mean",
    "num_cache_hits_mean",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "runtime_mean",
]

PAIR_FIELDNAMES = [
    "dataset",
    "seed",
    "feature_mode",
    "window_length",
    "stride",
    "r",
    "shots",
    "simulator",
    "affinity_normalization",
    "dtw_window",
    "test_index",
    "train_index",
    "test_sequence_id",
    "train_sequence_id",
    "test_label",
    "train_label",
    "exact_distance",
    "quantum_distance",
    "distance_abs_error",
    "cost_matrix_mae",
    "cost_matrix_rmse",
    "cost_matrix_max_abs_error",
    "num_local_costs",
    "git_commit",
    "timestamp",
]


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is currently implemented.")

    logger = get_logger("swap_local_sdtw")
    main_config = read_yaml(project_path(args.main_experiment_config))
    quantum_config = read_yaml(project_path(args.quantum_config))
    dataset_config = read_yaml(project_path(args.dataset_config))

    dataset = load_processed_dataset(_resolve_path(dataset_config["dataset"]["processed_dir"]))
    rows, pair_rows = run_experiment(
        dataset,
        seeds=args.seeds or [0, 1, 2],
        window_length=int(args.window_length),
        stride=int(args.stride),
        rank=int(args.rank),
        shots_values=args.shots or [512, 1024, 2048],
        simulator=args.simulator or quantum_config["quantum"].get("simulator", "sampling"),
        subset_enabled=_resolve_subset_flag(args.subset, quantum_config),
        train_per_class=int(args.train_per_class),
        test_per_class=int(args.test_per_class),
        limit_classes=args.limit_classes,
        feature_mode=args.feature_mode,
        center_sequence=bool(main_config["subspace"]["center_sequence"]),
        feature_standardization=args.feature_standardization,
        frame_l2_normalization=args.l2_normalize_frames,
        short_sequence_mode=args.short_sequence_mode,
        affinity_normalization=args.affinity_normalization,
        normalize_by_path_length=not args.no_path_length_normalization,
        dtw_window=args.dtw_window,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        cache_overlaps=not args.no_cache_overlaps,
        collect_pair_diagnostics=not args.no_pair_output,
        logger=logger,
    )

    output_path = _resolve_path(args.output)
    write_csv_rows(output_path, rows, fieldnames=RESULT_FIELDNAMES)
    logger.info("Wrote %s rows to %s.", len(rows), output_path)

    summary_rows = summarize_rows(rows)
    summary_path = _resolve_path(args.summary_output)
    write_csv_rows(summary_path, summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    logger.info("Wrote summary table to %s.", summary_path)

    if not args.no_pair_output:
        pair_output_path = _resolve_path(args.pair_output)
        write_csv_rows(pair_output_path, pair_rows, fieldnames=PAIR_FIELDNAMES)
        logger.info("Wrote %s pair diagnostic rows to %s.", len(pair_rows), pair_output_path)


def run_experiment(
    dataset: ProcessedDataset,
    *,
    seeds: list[int],
    window_length: int,
    stride: int,
    rank: int,
    shots_values: list[int],
    simulator: str,
    subset_enabled: bool,
    train_per_class: int,
    test_per_class: int,
    limit_classes: int | None,
    feature_mode: str,
    center_sequence: bool,
    feature_standardization: str,
    frame_l2_normalization: bool,
    short_sequence_mode: str,
    affinity_normalization: str,
    normalize_by_path_length: bool,
    dtw_window: Any,
    limit_train: int | None,
    limit_test: int | None,
    cache_overlaps: bool,
    collect_pair_diagnostics: bool,
    logger: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if rank >= window_length:
        raise ValueError("rank must be smaller than window_length for centered windows.")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode '{feature_mode}'. Known values: {FEATURE_MODES}.")
    if affinity_normalization not in AFFINITY_NORMALIZATIONS:
        raise ValueError(
            "Unknown affinity_normalization "
            f"'{affinity_normalization}'. Known values: {AFFINITY_NORMALIZATIONS}."
        )

    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    timestamp = utc_timestamp()
    git_commit = get_git_commit()
    dtw_label = window_ratio_label(dtw_window)

    for seed in [int(value) for value in seeds]:
        split = _load_seed_split(seed)
        train_indices, test_indices = _prepare_indices(
            dataset,
            split,
            subset_enabled=subset_enabled,
            train_per_class=train_per_class,
            test_per_class=test_per_class,
            limit_classes=limit_classes,
            limit_train=limit_train,
            limit_test=limit_test,
        )

        feature_start = perf_counter()
        train_local, train_kept, train_skipped = _compute_indexed_local_sequences(
            dataset,
            train_indices,
            rank=rank,
            window_length=window_length,
            stride=stride,
            center_sequence=center_sequence,
            feature_standardization=feature_standardization,
            frame_l2_normalization=frame_l2_normalization,
            feature_mode=feature_mode,
            short_sequence_mode=short_sequence_mode,
            logger=logger,
        )
        test_local, test_kept, test_skipped = _compute_indexed_local_sequences(
            dataset,
            test_indices,
            rank=rank,
            window_length=window_length,
            stride=stride,
            center_sequence=center_sequence,
            feature_standardization=feature_standardization,
            frame_l2_normalization=frame_l2_normalization,
            feature_mode=feature_mode,
            short_sequence_mode=short_sequence_mode,
            logger=logger,
        )
        feature_runtime = perf_counter() - feature_start
        if not train_local or not test_local:
            raise ValueError("Train and test local sequence collections must be non-empty.")

        y_train = dataset.labels[train_kept]
        y_test = dataset.labels[test_kept]
        train_windows_mean = float(np.mean([len(item.windows) for item in train_local]))
        test_windows_mean = float(np.mean([len(item.windows) for item in test_local]))

        exact_start = perf_counter()
        exact_distances = pairwise_local_sdtw_distances(
            test_local,
            train_local,
            metric="projection_affinity",
            affinity_normalization=affinity_normalization,
            normalize_by_path_length=normalize_by_path_length,
            window_ratio=dtw_window,
            backend="numpy",
        )
        exact_runtime = perf_counter() - exact_start
        exact_result = predict_1nn_from_distances(exact_distances, y_train)
        exact_accuracy = accuracy(y_test, exact_result.predictions)
        exact_f1 = macro_f1(y_test, exact_result.predictions)
        logger.info(
            "Exact Local-SDTW seed=%s L=%s stride=%s r=%s accuracy=%.4f macro_f1=%.4f train=%s test=%s.",
            seed,
            window_length,
            stride,
            rank,
            exact_accuracy,
            exact_f1,
            len(train_local),
            len(test_local),
        )

        for shots in [int(value) for value in shots_values]:
            logger.info(
                "Running SWAP Local-SDTW seed=%s L=%s stride=%s r=%s shots=%s train=%s test=%s.",
                seed,
                window_length,
                stride,
                rank,
                shots,
                len(train_local),
                len(test_local),
            )
            estimator = SwapTestOverlapEstimator(
                shots=shots,
                simulator=simulator,
                seed=_setting_seed(seed=seed, rank=rank, shots=shots),
                cache=cache_overlaps,
            )
            quantum_start = perf_counter()
            comparison = pairwise_quantum_exact_local_sdtw_comparison(
                test_local,
                train_local,
                estimator=estimator,
                affinity_normalization=affinity_normalization,
                normalize_by_path_length=normalize_by_path_length,
                window_ratio=dtw_window,
            )
            quantum_runtime = perf_counter() - quantum_start
            quantum_result = predict_1nn_from_distances(
                comparison.quantum_distances,
                y_train,
            )
            row = _result_row(
                dataset=dataset,
                seed=seed,
                feature_mode=feature_mode,
                window_length=window_length,
                stride=stride,
                rank=rank,
                shots=shots,
                simulator=simulator,
                affinity_normalization=affinity_normalization,
                subset_enabled=subset_enabled,
                train_per_class=train_per_class,
                test_per_class=test_per_class,
                dtw_window=dtw_label,
                normalize_by_path_length=normalize_by_path_length,
                y_test=y_test,
                exact_result=exact_result,
                quantum_result=quantum_result,
                exact_distances=exact_distances,
                comparison=comparison,
                feature_runtime=feature_runtime,
                exact_runtime=exact_runtime,
                quantum_runtime=quantum_runtime,
                train_size=len(train_local),
                test_size=len(test_local),
                skipped_sequences=train_skipped + test_skipped,
                train_windows_mean=train_windows_mean,
                test_windows_mean=test_windows_mean,
                num_overlap_estimates=estimator.num_estimates,
                num_cache_hits=estimator.num_cache_hits,
                git_commit=git_commit,
                timestamp=timestamp,
            )
            rows.append(row)
            if collect_pair_diagnostics:
                pair_rows.extend(
                    _pair_rows(
                        dataset=dataset,
                        seed=seed,
                        feature_mode=feature_mode,
                        window_length=window_length,
                        stride=stride,
                        rank=rank,
                        shots=shots,
                        simulator=simulator,
                        affinity_normalization=affinity_normalization,
                        dtw_window=dtw_label,
                        y_test=y_test,
                        y_train=y_train,
                        comparison=comparison,
                        git_commit=git_commit,
                        timestamp=timestamp,
                    )
                )
            logger.info(
                "seed=%s shots=%s exact=%.4f quantum=%.4f distance_mae=%.4f cost_mae=%.4f nn_agree=%.4f",
                seed,
                shots,
                row["exact_accuracy"],
                row["quantum_accuracy"],
                row["mean_abs_distance_error"],
                row["mean_cost_matrix_mae"],
                row["nearest_neighbor_agreement"],
            )

    return rows, pair_rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["method"]),
            str(row["feature_mode"]),
            int(row["window_length"]),
            int(row["stride"]),
            int(row["r"]),
            int(row["shots"]),
            str(row["overlap_method"]),
            str(row["simulator"]),
            str(row["affinity_normalization"]),
            _as_bool(row["subset"]),
            _optional_int(row["train_per_class"]),
            _optional_int(row["test_per_class"]),
            str(row["dtw_window"]),
            _as_bool(row["normalize_by_path_length"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        (
            dataset,
            method,
            feature_mode,
            window_length,
            stride,
            rank,
            shots,
            overlap_method,
            simulator,
            affinity_normalization,
            subset,
            train_per_class,
            test_per_class,
            dtw_window,
            normalized,
        ) = key
        exact_accuracy = _values(group_rows, "exact_accuracy")
        quantum_accuracy = _values(group_rows, "quantum_accuracy")
        exact_f1 = _values(group_rows, "exact_macro_f1")
        quantum_f1 = _values(group_rows, "quantum_macro_f1")
        total_runtime = _values(group_rows, "total_runtime_sec")
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "feature_mode": feature_mode,
                "window_length": window_length,
                "stride": stride,
                "r": rank,
                "shots": shots,
                "overlap_method": overlap_method,
                "simulator": simulator,
                "affinity_normalization": affinity_normalization,
                "subset": subset,
                "train_per_class": train_per_class if subset else "",
                "test_per_class": test_per_class if subset else "",
                "dtw_window": dtw_window,
                "normalize_by_path_length": normalized,
                "exact_accuracy_mean": _mean(exact_accuracy),
                "exact_accuracy_std": _std(exact_accuracy),
                "quantum_accuracy_mean": _mean(quantum_accuracy),
                "quantum_accuracy_std": _std(quantum_accuracy),
                "accuracy_gap_mean": _mean(
                    _values(group_rows, "accuracy_gap_quantum_minus_exact")
                ),
                "accuracy_gap_std": _std(
                    _values(group_rows, "accuracy_gap_quantum_minus_exact")
                ),
                "exact_macro_f1_mean": _mean(exact_f1),
                "exact_macro_f1_std": _std(exact_f1),
                "quantum_macro_f1_mean": _mean(quantum_f1),
                "quantum_macro_f1_std": _std(quantum_f1),
                "macro_f1_gap_mean": _mean(
                    _values(group_rows, "macro_f1_gap_quantum_minus_exact")
                ),
                "macro_f1_gap_std": _std(
                    _values(group_rows, "macro_f1_gap_quantum_minus_exact")
                ),
                "mean_abs_distance_error_mean": _mean(
                    _values(group_rows, "mean_abs_distance_error")
                ),
                "mean_abs_distance_error_std": _std(
                    _values(group_rows, "mean_abs_distance_error")
                ),
                "rmse_distance_error_mean": _mean(
                    _values(group_rows, "rmse_distance_error")
                ),
                "rmse_distance_error_std": _std(
                    _values(group_rows, "rmse_distance_error")
                ),
                "max_abs_distance_error_mean": _mean(
                    _values(group_rows, "max_abs_distance_error")
                ),
                "mean_cost_matrix_mae_mean": _mean(
                    _values(group_rows, "mean_cost_matrix_mae")
                ),
                "mean_cost_matrix_rmse_mean": _mean(
                    _values(group_rows, "mean_cost_matrix_rmse")
                ),
                "max_cost_matrix_mae_mean": _mean(
                    _values(group_rows, "max_cost_matrix_mae")
                ),
                "nearest_neighbor_agreement_mean": _mean(
                    _values(group_rows, "nearest_neighbor_agreement")
                ),
                "nearest_neighbor_agreement_std": _std(
                    _values(group_rows, "nearest_neighbor_agreement")
                ),
                "prediction_agreement_mean": _mean(
                    _values(group_rows, "prediction_agreement")
                ),
                "prediction_agreement_std": _std(
                    _values(group_rows, "prediction_agreement")
                ),
                "feature_runtime_mean": _mean(_values(group_rows, "feature_runtime_sec")),
                "exact_runtime_mean": _mean(_values(group_rows, "exact_runtime_sec")),
                "quantum_runtime_mean": _mean(_values(group_rows, "quantum_runtime_sec")),
                "total_runtime_mean": _mean(total_runtime),
                "train_size_mean": _mean(_values(group_rows, "train_size")),
                "test_size_mean": _mean(_values(group_rows, "test_size")),
                "num_train_windows_mean": _mean(
                    _values(group_rows, "num_train_windows_mean")
                ),
                "num_test_windows_mean": _mean(
                    _values(group_rows, "num_test_windows_mean")
                ),
                "num_dtw_pairs_mean": _mean(_values(group_rows, "num_dtw_pairs")),
                "num_local_costs_mean": _mean(_values(group_rows, "num_local_costs")),
                "num_overlap_estimates_mean": _mean(
                    _values(group_rows, "num_overlap_estimates")
                ),
                "num_cache_hits_mean": _mean(_values(group_rows, "num_cache_hits")),
                "accuracy_mean": _mean(quantum_accuracy),
                "accuracy_std": _std(quantum_accuracy),
                "macro_f1_mean": _mean(quantum_f1),
                "macro_f1_std": _std(quantum_f1),
                "runtime_mean": _mean(total_runtime),
            }
        )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msr_action3d"])
    parser.add_argument("--main-experiment-config", default="configs/experiment_main.yaml")
    parser.add_argument("--quantum-config", default="configs/experiment_quantum.yaml")
    parser.add_argument("--dataset-config", default="configs/msr_action3d.yaml")
    parser.add_argument("--feature-mode", default="position", choices=FEATURE_MODES)
    parser.add_argument("--window-length", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--shots", nargs="+", type=int, default=None)
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
        "--dtw-window",
        default="0.1",
        help="Use 'none' for unconstrained DTW, or a numeric Sakoe-Chiba ratio.",
    )
    parser.add_argument(
        "--subset",
        nargs="?",
        const="true",
        default="true",
        help="Boolean flag. Accepts true/false; bare --subset means true.",
    )
    parser.add_argument("--train-per-class", type=int, default=3)
    parser.add_argument("--test-per-class", type=int, default=2)
    parser.add_argument("--limit-classes", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--feature-standardization", default="none")
    parser.add_argument("--l2-normalize-frames", action="store_true")
    parser.add_argument("--short-sequence-mode", default="single", choices=["single", "error"])
    parser.add_argument("--no-path-length-normalization", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--no-cache-overlaps", action="store_true")
    parser.add_argument("--no-pair-output", action="store_true")
    parser.add_argument(
        "--output",
        default="results/raw/swap_local_sdtw_comparison_msr_action3d.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/tables/swap_local_sdtw_comparison_summary_msr_action3d.csv",
    )
    parser.add_argument(
        "--pair-output",
        default="results/raw/swap_local_sdtw_pair_diagnostics_msr_action3d.csv",
    )
    return parser.parse_args()


def _result_row(
    *,
    dataset: ProcessedDataset,
    seed: int,
    feature_mode: str,
    window_length: int,
    stride: int,
    rank: int,
    shots: int,
    simulator: str,
    affinity_normalization: str,
    subset_enabled: bool,
    train_per_class: int,
    test_per_class: int,
    dtw_window: str,
    normalize_by_path_length: bool,
    y_test: np.ndarray,
    exact_result: KNNResult,
    quantum_result: KNNResult,
    exact_distances: np.ndarray,
    comparison: QuantumLocalSdtwComparison,
    feature_runtime: float,
    exact_runtime: float,
    quantum_runtime: float,
    train_size: int,
    test_size: int,
    skipped_sequences: int,
    train_windows_mean: float,
    test_windows_mean: float,
    num_overlap_estimates: int,
    num_cache_hits: int,
    git_commit: str,
    timestamp: str,
) -> dict[str, Any]:
    exact_accuracy = accuracy(y_test, exact_result.predictions)
    quantum_accuracy = accuracy(y_test, quantum_result.predictions)
    exact_f1 = macro_f1(y_test, exact_result.predictions)
    quantum_f1 = macro_f1(y_test, quantum_result.predictions)
    distance_error = comparison.quantum_distances - exact_distances
    pair_cost_mae = np.asarray(
        [item.cost_matrix_mae for item in comparison.pair_diagnostics],
        dtype=np.float64,
    )
    pair_cost_rmse = np.asarray(
        [item.cost_matrix_rmse for item in comparison.pair_diagnostics],
        dtype=np.float64,
    )
    return {
        "dataset": dataset.dataset,
        "seed": seed,
        "method": "swap_local_sdtw",
        "feature_mode": feature_mode,
        "window_length": window_length,
        "stride": stride,
        "r": rank,
        "shots": shots,
        "overlap_method": "swap_test",
        "simulator": simulator,
        "affinity_normalization": affinity_normalization,
        "subset": subset_enabled,
        "train_per_class": train_per_class if subset_enabled else "",
        "test_per_class": test_per_class if subset_enabled else "",
        "dtw_window": dtw_window,
        "normalize_by_path_length": normalize_by_path_length,
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
        "mean_cost_matrix_mae": round(float(np.mean(pair_cost_mae)), 6),
        "mean_cost_matrix_rmse": round(float(np.mean(pair_cost_rmse)), 6),
        "max_cost_matrix_mae": round(float(np.max(pair_cost_mae)), 6),
        "nearest_neighbor_agreement": round(
            float(np.mean(exact_result.nearest_indices == quantum_result.nearest_indices)),
            6,
        ),
        "prediction_agreement": round(
            float(np.mean(exact_result.predictions == quantum_result.predictions)),
            6,
        ),
        "feature_runtime_sec": round(feature_runtime, 6),
        "exact_runtime_sec": round(exact_runtime, 6),
        "quantum_runtime_sec": round(quantum_runtime, 6),
        "total_runtime_sec": round(
            feature_runtime + exact_runtime + quantum_runtime,
            6,
        ),
        "train_size": train_size,
        "test_size": test_size,
        "skipped_sequences": skipped_sequences,
        "num_train_windows_mean": round(train_windows_mean, 6),
        "num_test_windows_mean": round(test_windows_mean, 6),
        "num_dtw_pairs": comparison.num_dtw_pairs,
        "num_local_costs": comparison.num_local_costs,
        "num_overlap_estimates": num_overlap_estimates,
        "num_cache_hits": num_cache_hits,
        "git_commit": git_commit,
        "timestamp": timestamp,
    }


def _pair_rows(
    *,
    dataset: ProcessedDataset,
    seed: int,
    feature_mode: str,
    window_length: int,
    stride: int,
    rank: int,
    shots: int,
    simulator: str,
    affinity_normalization: str,
    dtw_window: str,
    y_test: np.ndarray,
    y_train: np.ndarray,
    comparison: QuantumLocalSdtwComparison,
    git_commit: str,
    timestamp: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic in comparison.pair_diagnostics:
        row = asdict(diagnostic)
        row.update(
            {
                "dataset": dataset.dataset,
                "seed": seed,
                "feature_mode": feature_mode,
                "window_length": window_length,
                "stride": stride,
                "r": rank,
                "shots": shots,
                "simulator": simulator,
                "affinity_normalization": affinity_normalization,
                "dtw_window": dtw_window,
                "test_label": int(y_test[diagnostic.test_index]),
                "train_label": int(y_train[diagnostic.train_index]),
                "exact_distance": round(diagnostic.exact_distance, 6),
                "quantum_distance": round(diagnostic.quantum_distance, 6),
                "distance_abs_error": round(diagnostic.distance_abs_error, 6),
                "cost_matrix_mae": round(diagnostic.cost_matrix_mae, 6),
                "cost_matrix_rmse": round(diagnostic.cost_matrix_rmse, 6),
                "cost_matrix_max_abs_error": round(
                    diagnostic.cost_matrix_max_abs_error,
                    6,
                ),
                "git_commit": git_commit,
                "timestamp": timestamp,
            }
        )
        rows.append({field: row[field] for field in PAIR_FIELDNAMES})
    return rows


def _prepare_indices(
    dataset: ProcessedDataset,
    split: dict[str, Any],
    *,
    subset_enabled: bool,
    train_per_class: int,
    test_per_class: int,
    limit_classes: int | None,
    limit_train: int | None,
    limit_test: int | None,
) -> tuple[list[int], list[int]]:
    train_indices = [int(index) for index in split["train_indices"]]
    test_indices = [int(index) for index in split["test_indices"]]
    if limit_classes is not None:
        train_indices = _limit_classes(train_indices, dataset.labels, limit_classes)
        test_indices = _limit_classes(test_indices, dataset.labels, limit_classes)
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


def _compute_indexed_local_sequences(
    dataset: ProcessedDataset,
    indices: list[int],
    *,
    rank: int,
    window_length: int,
    stride: int,
    center_sequence: bool,
    feature_standardization: str,
    frame_l2_normalization: bool,
    feature_mode: str,
    short_sequence_mode: str,
    logger: Any,
) -> tuple[list[Any], list[int], int]:
    local_sequences: list[Any] = []
    kept_indices: list[int] = []
    skipped = 0
    for index in indices:
        try:
            local_sequence = compute_local_subspace_sequence(
                dataset.sequences[index],
                sequence_id=dataset.sequence_ids[index],
                rank=rank,
                window_length=window_length,
                stride=stride,
                center_sequence=center_sequence,
                feature_standardization=feature_standardization,
                frame_l2_normalization=frame_l2_normalization,
                feature_mode=feature_mode,
                short_sequence_mode=short_sequence_mode,
            )
        except ValueError as exc:
            skipped += 1
            logger.debug(
                "Skipping sequence %s for quantum Local-SDTW setting: %s",
                dataset.sequence_ids[index],
                exc,
            )
            continue
        local_sequences.append(local_sequence)
        kept_indices.append(index)
    return local_sequences, kept_indices, skipped


def _select_per_class(indices: list[int], labels: np.ndarray, per_class: int) -> list[int]:
    if per_class <= 0:
        raise ValueError("per_class must be positive.")
    selected: list[int] = []
    values = labels.astype(int)
    for label in sorted({int(values[index]) for index in indices}):
        selected.extend([int(index) for index in indices if int(values[index]) == label][:per_class])
    return selected


def _limit_classes(indices: list[int], labels: np.ndarray, num_classes: int) -> list[int]:
    if num_classes <= 0:
        raise ValueError("limit_classes must be positive.")
    values = labels.astype(int)
    allowed = set(sorted({int(values[index]) for index in indices})[:num_classes])
    return [int(index) for index in indices if int(values[index]) in allowed]


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _parse_bool(str(value))


def _optional_int(value: Any) -> int | str:
    if value in {"", None}:
        return ""
    return int(value)


def _setting_seed(*, seed: int, rank: int, shots: int) -> int:
    return int(seed * 1_000_003 + rank * 10_007 + shots)


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
