#!/usr/bin/env python
"""Run learned and kernel classical baselines."""

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

from src.baselines.gak import GAKBaseline
from src.baselines.kdtw import KDTWBaseline
from src.baselines.lstm import LSTMBaseline
from src.baselines.mlp import MLPBaseline
from src.baselines.random_forest import RandomForestBaseline
from src.data.validation import ProcessedDataset, load_processed_dataset
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import RESULT_FIELDNAMES, ResultRecord
from src.features.motion_features import apply_feature_mode
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


CLASSICAL_METHODS = ("mlp", "lstm", "random_forest", "kdtw", "gak")
FEATURE_MODES = (
    "position",
    "velocity",
    "acceleration",
    "position_velocity",
    "bone_vectors",
    "bone_velocity",
)


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is currently implemented.")

    logger = get_logger("classical_baselines")
    main_config = read_yaml(project_path(args.experiment_config))
    dataset_config = read_yaml(project_path(args.dataset_config))
    seeds = args.seeds or main_config["seeds"]
    methods = _validate_methods(args.methods)
    feature_modes = _validate_feature_modes(args.feature_modes)

    dataset = load_processed_dataset(
        _resolve_path(dataset_config["dataset"]["processed_dir"])
    )
    rows = run_experiment(
        dataset,
        dataset_config=dataset_config,
        methods=methods,
        seeds=[int(seed) for seed in seeds],
        feature_modes=feature_modes,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        args=args,
        logger=logger,
    )

    output_path = _resolve_path(args.output)
    write_csv_rows(output_path, rows, fieldnames=RESULT_FIELDNAMES)
    logger.info("Wrote %s rows to %s.", len(rows), output_path)


def run_experiment(
    dataset: ProcessedDataset,
    *,
    dataset_config: dict[str, Any],
    methods: list[str],
    seeds: list[int],
    feature_modes: list[str],
    limit_train: int | None,
    limit_test: int | None,
    args: argparse.Namespace,
    logger: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

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
        y_train = dataset.labels[train_indices]
        y_test = dataset.labels[test_indices]

        for feature_mode in feature_modes:
            train_sequences = [
                apply_feature_mode(sequence, feature_mode, dataset_config)
                for sequence in raw_train
            ]
            test_sequences = [
                apply_feature_mode(sequence, feature_mode, dataset_config)
                for sequence in raw_test
            ]

            for method in methods:
                logger.info(
                    "Running %s seed=%s feature=%s train=%s test=%s.",
                    method,
                    seed,
                    feature_mode,
                    len(train_sequences),
                    len(test_sequences),
                )
                row = _run_one_setting(
                    dataset_name=dataset.dataset,
                    seed=seed,
                    feature_mode=feature_mode,
                    method=method,
                    train_sequences=train_sequences,
                    test_sequences=test_sequences,
                    y_train=y_train,
                    y_test=y_test,
                    args=args,
                )
                logger.info(
                    "method=%s seed=%s feature=%s accuracy=%.4f macro_f1=%.4f "
                    "runtime=%.2fs",
                    row["method"],
                    row["seed"],
                    row["feature_mode"],
                    float(row["accuracy"]),
                    float(row["macro_f1"]),
                    float(row["runtime_sec"]),
                )
                rows.append(row)

    return rows


def _run_one_setting(
    *,
    dataset_name: str,
    seed: int,
    feature_mode: str,
    method: str,
    train_sequences: list[np.ndarray],
    test_sequences: list[np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    estimator, parameters = _make_estimator(method, seed=seed, args=args)
    start = perf_counter()
    estimator.fit(train_sequences, y_train)
    predictions = estimator.predict(test_sequences)
    runtime = perf_counter() - start

    record = ResultRecord(
        dataset=dataset_name,
        method=method,
        feature_mode=feature_mode,
        seed=seed,
        parameters=parameters,
        accuracy=round(accuracy(y_test, predictions), 6),
        macro_f1=round(macro_f1(y_test, predictions), 6),
        runtime_sec=round(runtime, 6),
    )
    return record.to_csv_row()


def _make_estimator(
    method: str,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    if method == "mlp":
        parameters = {
            "hidden_dim": int(args.mlp_hidden_dim),
            "dropout": float(args.mlp_dropout),
            "learning_rate": float(args.learning_rate),
            "max_epochs": int(args.max_epochs),
            "batch_size": int(args.batch_size),
            "validation_fraction": float(args.validation_fraction),
            "patience": int(args.patience),
            "device": str(args.device),
            "standardize": not bool(args.no_standardize),
        }
        return (
            MLPBaseline(seed=seed, **parameters),
            parameters,
        )

    if method == "lstm":
        parameters = {
            "hidden_dim": int(args.lstm_hidden_dim),
            "num_layers": int(args.lstm_num_layers),
            "learning_rate": float(args.learning_rate),
            "max_epochs": int(args.max_epochs),
            "batch_size": int(args.batch_size),
            "validation_fraction": float(args.validation_fraction),
            "patience": int(args.patience),
            "device": str(args.device),
            "standardize": not bool(args.no_standardize),
        }
        return (
            LSTMBaseline(seed=seed, **parameters),
            parameters,
        )

    if method == "random_forest":
        parameters = {
            "n_estimators": int(args.rf_n_estimators),
            "max_depth": args.rf_max_depth,
            "n_jobs": args.rf_n_jobs,
            "max_correlation_units": int(args.rf_max_correlation_units),
        }
        return (
            RandomForestBaseline(seed=seed, **parameters),
            parameters,
        )

    if method == "kdtw":
        parameters = {
            "gamma": float(args.kdtw_gamma),
            "normalize": not bool(args.no_kernel_normalization),
        }
        return KDTWBaseline(**parameters), parameters

    if method == "gak":
        auto_sigma = bool(args.gak_auto_sigma)
        parameters = {
            "sigma": float(args.gak_sigma),
            "normalize": not bool(args.no_kernel_normalization),
            "auto_sigma": auto_sigma,
        }
        return GAKBaseline(**parameters), parameters

    raise ValueError(f"Unknown classical baseline method '{method}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msr_action3d"])
    parser.add_argument("--experiment-config", default="configs/experiment_main.yaml")
    parser.add_argument("--dataset-config", default="configs/msr_action3d.yaml")
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=CLASSICAL_METHODS,
        help="One or more classical baselines to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--feature-modes",
        "--feature-mode",
        nargs="+",
        default=["position"],
        choices=FEATURE_MODES,
        help="One or more feature modes to evaluate.",
    )
    parser.add_argument("--output", default="results/raw/classical_baselines.csv")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)

    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-dropout", type=float, default=0.3)
    parser.add_argument("--lstm-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-num-layers", type=int, default=1)

    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-n-jobs", type=int, default=None)
    parser.add_argument("--rf-max-correlation-units", type=int, default=10)

    parser.add_argument("--kdtw-gamma", type=float, default=1.0)
    parser.add_argument("--gak-sigma", type=float, default=1.0)
    parser.add_argument(
        "--gak-auto-sigma",
        action="store_true",
        help="Estimate GAK sigma from training data using the median pairwise frame distance heuristic.",
    )
    parser.add_argument("--no-kernel-normalization", action="store_true")
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
    unknown = sorted(set(normalized) - set(CLASSICAL_METHODS))
    if unknown:
        raise ValueError(f"Unknown classical baseline methods: {unknown}.")
    return normalized


def _validate_feature_modes(feature_modes: list[str]) -> list[str]:
    normalized = [mode.lower() for mode in feature_modes]
    unknown = sorted(set(normalized) - set(FEATURE_MODES))
    if unknown:
        raise ValueError(f"Unknown feature modes: {unknown}.")
    return normalized


if __name__ == "__main__":
    main()
