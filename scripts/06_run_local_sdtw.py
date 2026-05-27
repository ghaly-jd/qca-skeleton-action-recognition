#!/usr/bin/env python
"""Run exact Local Subspace-DTW 1-NN experiments."""

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
from src.distances.dtw import window_ratio_label
from src.distances.local_sdtw import (
    LOCAL_SDTW_BACKENDS,
    LOCAL_SUBSPACE_DISTANCE_NAMES,
    pairwise_local_sdtw_distances,
)
from src.eval.knn import predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import get_git_commit, utc_timestamp
from src.features.local_subspace import FEATURE_MODES, compute_local_subspace_sequence
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


RESULT_FIELDNAMES = [
    "dataset",
    "seed",
    "method",
    "backend",
    "device",
    "dtype",
    "feature_mode",
    "window_length",
    "stride",
    "r",
    "local_distance",
    "dtw_window",
    "normalize_by_path_length",
    "accuracy",
    "macro_f1",
    "runtime_sec",
    "feature_runtime_sec",
    "distance_runtime_sec",
    "train_size",
    "test_size",
    "skipped_sequences",
    "num_train_windows_mean",
    "num_test_windows_mean",
    "git_commit",
    "timestamp",
]

SUMMARY_FIELDNAMES = [
    "dataset",
    "method",
    "backend",
    "device",
    "dtype",
    "feature_mode",
    "window_length",
    "stride",
    "r",
    "local_distance",
    "dtw_window",
    "normalize_by_path_length",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "runtime_mean",
    "feature_runtime_mean",
    "distance_runtime_mean",
    "num_train_windows_mean",
    "num_test_windows_mean",
]


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is currently implemented.")

    logger = get_logger("local_sdtw")
    main_config = read_yaml(project_path(args.main_experiment_config))
    local_config = read_yaml(project_path(args.local_sdtw_config))
    dataset_config = read_yaml(project_path(args.dataset_config))
    settings = local_config["local_sdtw"]

    feature_modes = args.feature_mode or settings["feature_modes"]
    window_lengths = args.window_lengths or settings["window_lengths"]
    strides = args.strides or settings["strides"]
    r_values = args.r_values or settings["r_values"]
    distances = args.distance or settings["distances"]
    dtw_windows = args.dtw_windows or settings["dtw_window_ratios"]
    seeds = args.seeds or main_config["seeds"]
    _validate_feature_modes(feature_modes)
    _validate_distances(distances)

    dataset = load_processed_dataset(_resolve_path(dataset_config["dataset"]["processed_dir"]))
    rows = run_experiment(
        dataset,
        seeds=[int(seed) for seed in seeds],
        feature_modes=[str(mode) for mode in feature_modes],
        window_lengths=[int(value) for value in window_lengths],
        strides=[int(value) for value in strides],
        r_values=[int(value) for value in r_values],
        distances=[str(distance) for distance in distances],
        dtw_windows=dtw_windows,
        normalize_by_path_length=(
            bool(settings["normalize_by_path_length"])
            and not args.no_path_length_normalization
        ),
        center_sequence=bool(main_config["subspace"]["center_sequence"]),
        feature_standardization=args.feature_standardization,
        frame_l2_normalization=args.l2_normalize_frames,
        short_sequence_mode=str(settings.get("short_sequence_mode", "single")),
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
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
    feature_modes: list[str],
    window_lengths: list[int],
    strides: list[int],
    r_values: list[int],
    distances: list[str],
    dtw_windows: list[Any],
    normalize_by_path_length: bool,
    center_sequence: bool,
    feature_standardization: str,
    frame_l2_normalization: bool,
    short_sequence_mode: str,
    backend: str,
    device: str | None,
    torch_dtype: str,
    limit_train: int | None,
    limit_test: int | None,
    logger: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamp = utc_timestamp()
    git_commit = get_git_commit()

    for seed in seeds:
        split = _load_seed_split(seed)
        train_indices_all = [int(index) for index in split["train_indices"]]
        test_indices_all = [int(index) for index in split["test_indices"]]
        if limit_train is not None:
            train_indices_all = train_indices_all[:limit_train]
        if limit_test is not None:
            test_indices_all = test_indices_all[:limit_test]

        for feature_mode in feature_modes:
            for window_length in window_lengths:
                for stride in strides:
                    for rank in r_values:
                        if rank >= window_length:
                            logger.info(
                                "Skipping invalid setting window_length=%s rank=%s.",
                                window_length,
                                rank,
                            )
                            continue

                        extract_start = perf_counter()
                        train_local, train_indices, train_skipped = _compute_indexed_local_sequences(
                            dataset,
                            train_indices_all,
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
                        test_local, test_indices, test_skipped = _compute_indexed_local_sequences(
                            dataset,
                            test_indices_all,
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
                        feature_runtime = perf_counter() - extract_start
                        if not train_local or not test_local:
                            logger.warning(
                                "Skipping empty setting seed=%s feature=%s L=%s stride=%s r=%s.",
                                seed,
                                feature_mode,
                                window_length,
                                stride,
                                rank,
                            )
                            continue

                        y_train = dataset.labels[train_indices]
                        y_test = dataset.labels[test_indices]
                        train_windows_mean = float(
                            np.mean([len(item.windows) for item in train_local])
                        )
                        test_windows_mean = float(
                            np.mean([len(item.windows) for item in test_local])
                        )

                        for distance in distances:
                            for dtw_window in dtw_windows:
                                label = window_ratio_label(dtw_window)
                                logger.info(
                                    "Running Local-SDTW seed=%s feature=%s L=%s stride=%s "
                                    "r=%s distance=%s dtw_window=%s backend=%s train=%s test=%s.",
                                    seed,
                                    feature_mode,
                                    window_length,
                                    stride,
                                    rank,
                                    distance,
                                    label,
                                    backend,
                                    len(train_local),
                                    len(test_local),
                                )
                                distance_start = perf_counter()
                                distance_matrix = pairwise_local_sdtw_distances(
                                    test_local,
                                    train_local,
                                    metric=distance,
                                    normalize_by_path_length=normalize_by_path_length,
                                    window_ratio=dtw_window,
                                    backend=backend,
                                    torch_device=device,
                                    torch_dtype=torch_dtype,
                                )
                                result = predict_1nn_from_distances(distance_matrix, y_train)
                                distance_runtime = perf_counter() - distance_start
                                runtime = feature_runtime + distance_runtime

                                row = {
                                    "dataset": dataset.dataset,
                                    "seed": seed,
                                    "method": "local_sdtw_exact",
                                    "backend": backend,
                                    "device": _device_label(backend, device),
                                    "dtype": torch_dtype if backend.startswith("torch") else "",
                                    "feature_mode": feature_mode,
                                    "window_length": window_length,
                                    "stride": stride,
                                    "r": rank,
                                    "local_distance": distance,
                                    "dtw_window": label,
                                    "normalize_by_path_length": normalize_by_path_length,
                                    "accuracy": round(accuracy(y_test, result.predictions), 6),
                                    "macro_f1": round(macro_f1(y_test, result.predictions), 6),
                                    "runtime_sec": round(runtime, 6),
                                    "feature_runtime_sec": round(feature_runtime, 6),
                                    "distance_runtime_sec": round(distance_runtime, 6),
                                    "train_size": len(train_local),
                                    "test_size": len(test_local),
                                    "skipped_sequences": train_skipped + test_skipped,
                                    "num_train_windows_mean": round(train_windows_mean, 6),
                                    "num_test_windows_mean": round(test_windows_mean, 6),
                                    "git_commit": git_commit,
                                    "timestamp": timestamp,
                                }
                                rows.append(row)
                                logger.info(
                                    "seed=%s L=%s stride=%s r=%s distance=%s window=%s "
                                    "accuracy=%.4f macro_f1=%.4f runtime=%.2fs",
                                    seed,
                                    window_length,
                                    stride,
                                    rank,
                                    distance,
                                    label,
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
            str(row["method"]),
            str(row.get("backend", "")),
            str(row.get("device", "")),
            str(row.get("dtype", "")),
            str(row["feature_mode"]),
            int(row["window_length"]),
            int(row["stride"]),
            int(row["r"]),
            str(row["local_distance"]),
            str(row["dtw_window"]),
            bool(row["normalize_by_path_length"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (
        dataset,
        method,
        backend,
        device,
        dtype,
        feature_mode,
        window_length,
        stride,
        rank,
        distance,
        dtw_window,
        normalized,
    ), group_rows in sorted(groups.items()):
        accuracies = np.asarray([float(row["accuracy"]) for row in group_rows])
        macro_f1_values = np.asarray([float(row["macro_f1"]) for row in group_rows])
        runtimes = np.asarray([float(row["runtime_sec"]) for row in group_rows])
        feature_runtimes = np.asarray(
            [float(row["feature_runtime_sec"]) for row in group_rows]
        )
        distance_runtimes = np.asarray(
            [float(row["distance_runtime_sec"]) for row in group_rows]
        )
        train_windows = np.asarray(
            [float(row["num_train_windows_mean"]) for row in group_rows]
        )
        test_windows = np.asarray(
            [float(row["num_test_windows_mean"]) for row in group_rows]
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "backend": backend,
                "device": device,
                "dtype": dtype,
                "feature_mode": feature_mode,
                "window_length": window_length,
                "stride": stride,
                "r": rank,
                "local_distance": distance,
                "dtw_window": dtw_window,
                "normalize_by_path_length": normalized,
                "accuracy_mean": round(float(accuracies.mean()), 6),
                "accuracy_std": round(float(accuracies.std(ddof=0)), 6),
                "macro_f1_mean": round(float(macro_f1_values.mean()), 6),
                "macro_f1_std": round(float(macro_f1_values.std(ddof=0)), 6),
                "runtime_mean": round(float(runtimes.mean()), 6),
                "feature_runtime_mean": round(float(feature_runtimes.mean()), 6),
                "distance_runtime_mean": round(float(distance_runtimes.mean()), 6),
                "num_train_windows_mean": round(float(train_windows.mean()), 6),
                "num_test_windows_mean": round(float(test_windows.mean()), 6),
            }
        )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msr_action3d"])
    parser.add_argument("--main-experiment-config", default="configs/experiment_main.yaml")
    parser.add_argument("--local-sdtw-config", default="configs/experiment_local_sdtw.yaml")
    parser.add_argument("--dataset-config", default="configs/msr_action3d.yaml")
    parser.add_argument("--feature-mode", nargs="+", default=None, choices=FEATURE_MODES)
    parser.add_argument("--window-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--strides", nargs="+", type=int, default=None)
    parser.add_argument("--r-values", nargs="+", type=int, default=None)
    parser.add_argument(
        "--distance",
        nargs="+",
        default=None,
        choices=LOCAL_SUBSPACE_DISTANCE_NAMES,
    )
    parser.add_argument(
        "--dtw-windows",
        nargs="+",
        default=None,
        help="Use 'none' for unconstrained DTW, plus numeric Sakoe-Chiba ratios.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--feature-standardization", default="none")
    parser.add_argument("--l2-normalize-frames", action="store_true")
    parser.add_argument("--no-path-length-normalization", action="store_true")
    parser.add_argument(
        "--backend",
        default="numpy",
        choices=LOCAL_SDTW_BACKENDS,
        help="Distance backend. Use torch_cuda for CUDA projection-affinity Local-SDTW.",
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
        "--output",
        default="results/raw/local_subspace_dtw_msr_action3d.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/tables/local_subspace_dtw_summary_msr_action3d.csv",
    )
    return parser.parse_args()


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
                "Skipping sequence %s for local subspace setting: %s",
                dataset.sequence_ids[index],
                exc,
            )
            continue
        local_sequences.append(local_sequence)
        kept_indices.append(index)
    return local_sequences, kept_indices, skipped


def _load_seed_split(seed: int) -> dict[str, Any]:
    split_path = project_path("data", "splits", f"msr_cross_subject_seed{seed}.json")
    if not split_path.exists():
        split_path = project_path("data", "splits", "msr_cross_subject.json")
    return read_json(split_path)


def _validate_feature_modes(feature_modes: list[str]) -> None:
    unknown = sorted(set(feature_modes) - set(FEATURE_MODES))
    if unknown:
        raise ValueError(f"Unknown feature modes {unknown}. Known values: {FEATURE_MODES}.")


def _validate_distances(distances: list[str]) -> None:
    unknown = sorted(set(distances) - set(LOCAL_SUBSPACE_DISTANCE_NAMES))
    if unknown:
        known = ", ".join(LOCAL_SUBSPACE_DISTANCE_NAMES)
        raise ValueError(f"Unknown local distances {unknown}. Known values: {known}.")


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


def _device_label(backend: str, device: str | None) -> str:
    if backend == "torch_cuda" and device is None:
        return "cuda"
    if backend == "torch_cpu":
        return "cpu"
    return device or ""


if __name__ == "__main__":
    main()
