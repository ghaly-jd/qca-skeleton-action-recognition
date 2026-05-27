#!/usr/bin/env python
"""Run DTW 1-NN classification baselines."""

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
from src.distances.dtw import pairwise_dtw_distances, window_ratio_label
from src.eval.knn import predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import get_git_commit, utc_timestamp
from src.features.motion_features import apply_feature_mode
from src.features.pca_projection import fit_pca_from_sequences
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


RESULT_FIELDNAMES = [
    "dataset",
    "seed",
    "method",
    "feature_mode",
    "pca_k",
    "backend",
    "device",
    "dtype",
    "window_ratio",
    "normalize_by_path_length",
    "accuracy",
    "macro_f1",
    "runtime_sec",
    "train_size",
    "test_size",
    "git_commit",
    "timestamp",
]

SUMMARY_FIELDNAMES = [
    "dataset",
    "method",
    "pca_k",
    "backend",
    "device",
    "dtype",
    "window_ratio",
    "normalize_by_path_length",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "runtime_mean",
]


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is implemented for Phase 3.")

    logger = get_logger("dtw_baselines")
    experiment_config = read_yaml(project_path(args.experiment_config))
    dataset_config = read_yaml(project_path(args.dataset_config))

    seeds = args.seeds or experiment_config["seeds"]
    window_ratios = args.window_ratios
    if window_ratios is None:
        window_ratios = experiment_config["dtw"]["window_ratios"]
    pca_k_values = args.pca_k_values or experiment_config["pca_dtw"]["k_values"]

    feature_modes = args.feature_mode or ["position"]
    processed_dir = _resolve_path(dataset_config["dataset"]["processed_dir"])
    dataset = load_processed_dataset(processed_dir)
    rows = []
    for feature_mode in feature_modes:
        logger.info("Running feature_mode=%s", feature_mode)
        rows.extend(run_experiment(
            dataset,
            seeds=[int(seed) for seed in seeds],
            methods=args.methods,
            feature_mode=feature_mode,
            dataset_config=dataset_config,
            window_ratios=window_ratios,
            pca_k_values=[int(k) for k in pca_k_values],
            normalize_by_path_length=not args.no_path_length_normalization,
            backend=args.backend,
            device=args.device,
            dtype=args.torch_dtype,
            limit_train=args.limit_train,
            limit_test=args.limit_test,
            logger=logger,
        ))

    output_path = _resolve_path(args.output)
    write_csv_rows(output_path, rows, fieldnames=RESULT_FIELDNAMES)
    logger.info("Wrote %s rows to %s.", len(rows), output_path)

    summary_rows = summarize_rows(rows)
    summary_path = _resolve_path(args.summary_output)
    write_csv_rows(summary_path, summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    logger.info("Wrote summary table to %s.", summary_path)

    best = max(rows, key=lambda row: float(row["accuracy"]))
    logger.info(
        "Best DTW row: method=%s seed=%s window=%s accuracy=%.4f macro_f1=%.4f",
        best["method"],
        best["seed"],
        best["window_ratio"],
        float(best["accuracy"]),
        float(best["macro_f1"]),
    )


def run_experiment(
    dataset: ProcessedDataset,
    *,
    seeds: list[int],
    methods: list[str],
    feature_mode: str = "position",
    dataset_config: dict[str, Any] | None = None,
    window_ratios: list[Any],
    pca_k_values: list[int],
    normalize_by_path_length: bool,
    backend: str,
    device: str | None,
    dtype: str,
    limit_train: int | None,
    limit_test: int | None,
    logger: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamp = utc_timestamp()
    git_commit = get_git_commit()
    methods = _validate_methods(methods)

    for seed in seeds:
        split = _load_seed_split(seed)
        train_indices = [int(index) for index in split["train_indices"]]
        test_indices = [int(index) for index in split["test_indices"]]
        if limit_train is not None:
            train_indices = train_indices[:limit_train]
        if limit_test is not None:
            test_indices = test_indices[:limit_test]
        if not train_indices or not test_indices:
            raise ValueError("Train and test splits must both be non-empty.")

        raw_train = [dataset.sequences[index] for index in train_indices]
        raw_test = [dataset.sequences[index] for index in test_indices]
        train_sequences = [apply_feature_mode(X, feature_mode, dataset_config) for X in raw_train]
        test_sequences = [apply_feature_mode(X, feature_mode, dataset_config) for X in raw_test]
        y_train = dataset.labels[train_indices]
        y_test = dataset.labels[test_indices]

        if "raw_dtw" in methods:
            for window_ratio in window_ratios:
                rows.append(
                    _run_one_dtw_setting(
                        dataset_name=dataset.dataset,
                        seed=seed,
                        feature_mode=feature_mode,
                        method="raw_dtw",
                        pca_k="",
                        train_sequences=train_sequences,
                        test_sequences=test_sequences,
                        y_train=y_train,
                        y_test=y_test,
                        window_ratio=window_ratio,
                        normalize_by_path_length=normalize_by_path_length,
                        backend=backend,
                        device=device,
                        dtype=dtype,
                        extra_runtime_sec=0.0,
                        git_commit=git_commit,
                        timestamp=timestamp,
                        logger=logger,
                    )
                )

        if "pca_dtw" in methods:
            for pca_k in pca_k_values:
                pca_start = perf_counter()
                pca_model = fit_pca_from_sequences(
                    train_sequences,
                    n_components=pca_k,
                )
                train_projected = pca_model.transform_sequences(train_sequences)
                test_projected = pca_model.transform_sequences(test_sequences)
                pca_runtime = perf_counter() - pca_start
                explained = float(np.sum(pca_model.explained_variance_ratio))
                logger.info(
                    "Fitted PCA seed=%s k=%s explained_variance=%.4f runtime=%.2fs.",
                    seed,
                    pca_k,
                    explained,
                    pca_runtime,
                )

                for window_ratio in window_ratios:
                    rows.append(
                        _run_one_dtw_setting(
                            dataset_name=dataset.dataset,
                            seed=seed,
                            feature_mode=feature_mode,
                            method="pca_dtw",
                            pca_k=pca_k,
                            train_sequences=train_projected,
                            test_sequences=test_projected,
                            y_train=y_train,
                            y_test=y_test,
                            window_ratio=window_ratio,
                            normalize_by_path_length=normalize_by_path_length,
                            backend=backend,
                            device=device,
                            dtype=dtype,
                            extra_runtime_sec=pca_runtime,
                            git_commit=git_commit,
                            timestamp=timestamp,
                            logger=logger,
                        )
                    )

    return rows


def _run_one_dtw_setting(
    *,
    dataset_name: str,
    seed: int,
    feature_mode: str = "position",
    method: str,
    pca_k: int | str,
    train_sequences: list[np.ndarray],
    test_sequences: list[np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    window_ratio: Any,
    normalize_by_path_length: bool,
    backend: str,
    device: str | None,
    dtype: str,
    extra_runtime_sec: float,
    git_commit: str,
    timestamp: str,
    logger: Any,
) -> dict[str, Any]:
    label = window_ratio_label(window_ratio)
    method_label = method if pca_k == "" else f"{method}(k={pca_k})"
    logger.info(
        "Running %s seed=%s window=%s backend=%s train=%s test=%s.",
        method_label,
        seed,
        label,
        backend,
        len(train_sequences),
        len(test_sequences),
    )
    start = perf_counter()
    distance_matrix = pairwise_dtw_distances(
        test_sequences,
        train_sequences,
        normalize_by_path_length=normalize_by_path_length,
        window_ratio=window_ratio,
        backend=backend,
        torch_device=device,
        torch_dtype=dtype,
    )
    result = predict_1nn_from_distances(distance_matrix, y_train)
    runtime = (perf_counter() - start) + extra_runtime_sec

    row = {
        "dataset": dataset_name,
        "seed": seed,
        "method": method,
        "feature_mode": feature_mode,
        "pca_k": pca_k,
        "backend": backend,
        "device": _device_label(backend, device),
        "dtype": dtype if backend.startswith("torch") else "",
        "window_ratio": label,
        "normalize_by_path_length": normalize_by_path_length,
        "accuracy": round(accuracy(y_test, result.predictions), 6),
        "macro_f1": round(macro_f1(y_test, result.predictions), 6),
        "runtime_sec": round(runtime, 6),
        "train_size": len(train_sequences),
        "test_size": len(test_sequences),
        "git_commit": git_commit,
        "timestamp": timestamp,
    }
    logger.info(
        "method=%s seed=%s window=%s accuracy=%.4f macro_f1=%.4f runtime=%.2fs",
        method_label,
        seed,
        label,
        row["accuracy"],
        row["macro_f1"],
        row["runtime_sec"],
    )
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["method"]),
            str(row["pca_k"]),
            str(row["backend"]),
            str(row["device"]),
            str(row["dtype"]),
            str(row["window_ratio"]),
            bool(row["normalize_by_path_length"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (
        dataset,
        method,
        pca_k,
        backend,
        device,
        dtype,
        window_ratio,
        normalized,
    ), group_rows in sorted(groups.items()):
        accuracies = np.asarray([float(row["accuracy"]) for row in group_rows])
        macro_f1_values = np.asarray([float(row["macro_f1"]) for row in group_rows])
        runtimes = np.asarray([float(row["runtime_sec"]) for row in group_rows])
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "pca_k": pca_k,
                "backend": backend,
                "device": device,
                "dtype": dtype,
                "window_ratio": window_ratio,
                "normalize_by_path_length": normalized,
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
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw_dtw"],
        choices=["raw_dtw", "pca_dtw"],
    )
    parser.add_argument(
        "--window-ratios",
        nargs="+",
        default=None,
        help="Use 'none' for unconstrained DTW, plus numeric Sakoe-Chiba ratios.",
    )
    parser.add_argument("--pca-k-values", nargs="+", type=int, default=None)
    parser.add_argument("--no-path-length-normalization", action="store_true")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "numpy", "numba", "torch", "torch_cpu", "torch_cuda"],
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for torch/torch_cuda backends, e.g. cuda:0 or cuda:1.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Floating-point dtype for torch backends.",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
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
    parser.add_argument("--output", default="results/raw/dtw_baselines.csv")
    parser.add_argument(
        "--summary-output",
        default="results/tables/dtw_baselines_summary.csv",
    )
    return parser.parse_args()


def _load_seed_split(seed: int) -> dict[str, Any]:
    split_path = project_path("data", "splits", f"msr_cross_subject_seed{seed}.json")
    if not split_path.exists():
        split_path = project_path("data", "splits", "msr_cross_subject.json")
    return read_json(split_path)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


def _validate_methods(methods: list[str]) -> list[str]:
    normalized = [method.lower() for method in methods]
    unknown = sorted(set(normalized) - {"raw_dtw", "pca_dtw"})
    if unknown:
        raise ValueError(f"Unknown DTW baseline methods: {unknown}.")
    return normalized


def _device_label(backend: str, device: str | None) -> str:
    if backend == "torch_cuda" and device is None:
        return "cuda"
    if backend == "torch_cpu":
        return "cpu"
    return device or ""


if __name__ == "__main__":
    main()
