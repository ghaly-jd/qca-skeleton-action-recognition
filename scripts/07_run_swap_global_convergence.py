#!/usr/bin/env python
"""Run formal exact-vs-SWAP global subspace-affinity convergence experiments."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.data.validation import load_processed_dataset
from src.utils.io import read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


SUMMARY_FIELDNAMES = [
    "dataset",
    "r",
    "shots",
    "overlap_method",
    "simulator",
    "affinity_normalization",
    "subset",
    "train_per_class",
    "test_per_class",
    "train_size_mean",
    "test_size_mean",
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
    "max_abs_distance_error_std",
    "nearest_neighbor_agreement_mean",
    "nearest_neighbor_agreement_std",
    "prediction_agreement_mean",
    "prediction_agreement_std",
    "exact_runtime_mean",
    "quantum_runtime_mean",
    "total_runtime_mean",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "runtime_mean",
]


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is implemented for Phase 6.")

    comparison = _load_comparison_module()
    logger = get_logger("swap_global_convergence")

    quantum_config = read_yaml(project_path(args.experiment_config))
    main_config = read_yaml(project_path(args.main_experiment_config))
    dataset_config = read_yaml(project_path(args.dataset_config))
    convergence_config = quantum_config.get("swap_global_convergence", {})

    r_values = args.r_values or convergence_config.get("r_values") or [2, 3, 4]
    shots_values = (
        args.shots
        or convergence_config.get("shots")
        or quantum_config["subspace_affinity"].get("shots_full", [128, 256, 512, 1024])
    )
    seeds = args.seeds or main_config["seeds"]
    simulator = args.simulator or quantum_config["quantum"].get("simulator", "sampling")
    subset_enabled = comparison._resolve_subset_flag(args.subset, quantum_config)
    train_per_class = (
        args.train_per_class
        or convergence_config.get("train_per_class")
        or quantum_config["subset"]["train_per_class"]
    )
    test_per_class = (
        args.test_per_class
        or convergence_config.get("test_per_class")
        or quantum_config["subset"]["test_per_class"]
    )

    dataset = load_processed_dataset(
        _resolve_path(dataset_config["dataset"]["processed_dir"])
    )
    rows = comparison.run_comparison(
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
    write_csv_rows(output_path, rows, fieldnames=comparison.RESULT_FIELDNAMES)
    logger.info("Wrote %s raw rows to %s.", len(rows), output_path)

    summary_rows = summarize_rows(rows)
    summary_path = _resolve_path(args.summary_output)
    write_csv_rows(summary_path, summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    logger.info("Wrote summary table to %s.", summary_path)


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
            _as_bool(row["subset"]),
            _optional_int(row.get("train_per_class")),
            _optional_int(row.get("test_per_class")),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        (
            dataset,
            rank,
            shots,
            overlap_method,
            simulator,
            affinity_normalization,
            subset,
            train_per_class,
            test_per_class,
        ) = key

        exact_accuracy = _values(group_rows, "exact_accuracy")
        quantum_accuracy = _values(group_rows, "quantum_accuracy")
        exact_f1 = _values(group_rows, "exact_macro_f1")
        quantum_f1 = _values(group_rows, "quantum_macro_f1")
        total_runtime = _values(group_rows, "total_runtime_sec")

        summary_rows.append(
            {
                "dataset": dataset,
                "r": rank,
                "shots": shots,
                "overlap_method": overlap_method,
                "simulator": simulator,
                "affinity_normalization": affinity_normalization,
                "subset": subset,
                "train_per_class": train_per_class if subset else "",
                "test_per_class": test_per_class if subset else "",
                "train_size_mean": _mean(_values(group_rows, "train_size")),
                "test_size_mean": _mean(_values(group_rows, "test_size")),
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
                "max_abs_distance_error_std": _std(
                    _values(group_rows, "max_abs_distance_error")
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
                "exact_runtime_mean": _mean(_values(group_rows, "exact_runtime_sec")),
                "quantum_runtime_mean": _mean(
                    _values(group_rows, "quantum_runtime_sec")
                ),
                "total_runtime_mean": _mean(total_runtime),
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
        choices=["projection_frobenius", "mean"],
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
        default="results/raw/swap_global_exact_comparison_msr_action3d.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/tables/swap_global_exact_comparison_summary_msr_action3d.csv",
    )
    return parser.parse_args()


def _load_comparison_module() -> Any:
    path = project_path("scripts", "04_compare_quantum_exact_subset.py")
    spec = importlib.util.spec_from_file_location("quantum_exact_subset_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load comparison module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _mean(values: np.ndarray) -> float:
    return round(float(np.mean(values)), 6)


def _std(values: np.ndarray) -> float:
    return round(float(np.std(values, ddof=0)), 6)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_int(value: Any) -> int | str:
    if value in {"", None}:
        return ""
    return int(value)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


if __name__ == "__main__":
    main()
