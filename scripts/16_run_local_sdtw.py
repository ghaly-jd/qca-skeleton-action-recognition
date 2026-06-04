#!/usr/bin/env python
"""Run Exact and SWAP Local-SDTW 1-NN classification (Steps 1.19 and 1.20)."""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.data.validation import ProcessedDataset, load_processed_dataset
from src.distances.local_sdtw import (
    _extract_local_bases,
    dtw_distance_from_cost_matrix,
)
from src.eval.knn import predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import (
    RESULT_FIELDNAMES,
    ResultRecord,
    get_git_commit,
    utc_timestamp,
)
from src.features.motion_features import apply_feature_mode
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import project_path


BACKENDS = ("exact", "swap")


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is currently implemented.")

    logger = get_logger("local_sdtw_runner")
    main_config = read_yaml(project_path(args.experiment_config))
    dataset_config = read_yaml(project_path(args.dataset_config))
    seeds = args.seeds if args.seeds is not None else main_config["seeds"]
    feature_modes = args.feature_modes or ["position"]
    shots_list = args.shots or [2048]

    dataset = load_processed_dataset(
        _resolve_path(dataset_config["dataset"]["processed_dir"])
    )

    rows = run_experiment(
        dataset,
        dataset_config=dataset_config,
        backend=args.backend,
        window_sizes=args.window_sizes,
        strides=args.strides,
        ranks=args.ranks,
        seeds=[int(s) for s in seeds],
        feature_modes=feature_modes,
        shots_list=[int(s) for s in shots_list],
        dtw_window=args.dtw_window,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        logger=logger,
    )

    output_path = _resolve_path(args.output)
    write_csv_rows(output_path, rows, fieldnames=RESULT_FIELDNAMES)
    logger.info("Wrote %d rows to %s.", len(rows), output_path)


def run_experiment(
    dataset: ProcessedDataset,
    *,
    dataset_config: dict[str, Any],
    backend: str,
    window_sizes: list[int],
    strides: list[int],
    ranks: list[int],
    seeds: list[int],
    feature_modes: list[str],
    shots_list: list[int],
    dtw_window: str | None,
    limit_train: int | None,
    limit_test: int | None,
    logger: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    git_commit = get_git_commit()
    timestamp = utc_timestamp()
    dtw_window_val = _parse_window_ratio(dtw_window)

    for feature_mode in feature_modes:
        logger.info("feature_mode=%s", feature_mode)
        for seed in seeds:
            split = _load_seed_split(seed)
            train_indices = [int(i) for i in split["train_indices"]]
            test_indices = [int(i) for i in split["test_indices"]]
            if limit_train is not None:
                train_indices = train_indices[:limit_train]
            if limit_test is not None:
                test_indices = test_indices[:limit_test]

            raw_train = [dataset.sequences[i] for i in train_indices]
            raw_test = [dataset.sequences[i] for i in test_indices]
            train_seqs = [apply_feature_mode(X, feature_mode, dataset_config) for X in raw_train]
            test_seqs = [apply_feature_mode(X, feature_mode, dataset_config) for X in raw_test]
            y_train = dataset.labels[train_indices]
            y_test = dataset.labels[test_indices]

            for window_size, stride, rank in product(window_sizes, strides, ranks):
                logger.info(
                    "seed=%d window=%d stride=%d rank=%d backend=%s",
                    seed,
                    window_size,
                    stride,
                    rank,
                    backend,
                )
                try:
                    train_bases = [
                        _extract_local_bases(
                            X, window_size=window_size, stride=stride, rank=rank
                        )
                        for X in train_seqs
                    ]
                    test_bases = [
                        _extract_local_bases(
                            X, window_size=window_size, stride=stride, rank=rank
                        )
                        for X in test_seqs
                    ]
                except ValueError as exc:
                    logger.warning("Skipping window=%d stride=%d rank=%d: %s", window_size, stride, rank, exc)
                    continue

                shots_iter: list[int | None] = [None] if backend == "exact" else shots_list
                for shots in shots_iter:
                    start = perf_counter()
                    if backend == "exact":
                        dist_matrix = _pairwise_distances_exact(
                            test_bases,
                            train_bases,
                            rank=rank,
                            dtw_window=dtw_window_val,
                        )
                    else:
                        rng = np.random.default_rng(seed)
                        dist_matrix = _pairwise_distances_swap(
                            test_bases,
                            train_bases,
                            shots=shots,
                            rng=rng,
                            rank=rank,
                            dtw_window=dtw_window_val,
                        )
                    result = predict_1nn_from_distances(dist_matrix, y_train)
                    runtime = perf_counter() - start

                    params: dict[str, Any] = {
                        "window_size": window_size,
                        "stride": stride,
                        "rank": rank,
                        "dtw_window": dtw_window if dtw_window else "none",
                    }
                    if backend == "swap" and shots is not None:
                        params["shots"] = shots

                    acc = round(accuracy(y_test, result.predictions), 6)
                    f1 = round(macro_f1(y_test, result.predictions), 6)
                    record = ResultRecord(
                        dataset=dataset.dataset,
                        method=f"local_sdtw_{backend}",
                        feature_mode=feature_mode,
                        seed=seed,
                        parameters=params,
                        accuracy=acc,
                        macro_f1=f1,
                        runtime_sec=round(runtime, 6),
                        git_commit=git_commit,
                        timestamp=timestamp,
                    )
                    rows.append(record.to_csv_row())
                    logger.info(
                        "method=local_sdtw_%s feature=%s seed=%d w=%d s=%d r=%d%s "
                        "acc=%.4f f1=%.4f rt=%.2fs",
                        backend,
                        feature_mode,
                        seed,
                        window_size,
                        stride,
                        rank,
                        f" shots={shots}" if shots is not None else "",
                        acc,
                        f1,
                        runtime,
                    )
    return rows


def _pairwise_distances_exact(
    test_bases_list: list[np.ndarray],
    train_bases_list: list[np.ndarray],
    *,
    rank: int,
    dtw_window: Any,
) -> np.ndarray:
    """Vectorized exact Local-SDTW distance matrix.

    Uses einsum to compute all window-pair overlaps at once, replacing the
    O(M1*M2*rank^2) Python loop with a single numpy op per sequence pair.
    Matches the scalar _build_cost_matrix normalization: sum/rank.
    """
    n_test = len(test_bases_list)
    n_train = len(train_bases_list)
    distances = np.empty((n_test, n_train), dtype=np.float64)
    for i, test_bases in enumerate(test_bases_list):
        for j, train_bases in enumerate(train_bases_list):
            # overlaps[m,n,a,b] = test_bases[m,:,a] · train_bases[n,:,b]
            overlaps = np.einsum("mda,ndb->mnab", test_bases, train_bases)
            affinity = (overlaps ** 2).sum(axis=(-2, -1)) / rank  # (M1, M2)
            cost_matrix = np.clip(1.0 - affinity, 0.0, None)
            distances[i, j] = dtw_distance_from_cost_matrix(
                cost_matrix,
                normalize_by_path_length=True,
                window_ratio=dtw_window,
            )
    return distances


def _pairwise_distances_swap(
    test_bases_list: list[np.ndarray],
    train_bases_list: list[np.ndarray],
    *,
    shots: int,
    rng: np.random.Generator,
    rank: int,
    dtw_window: Any,
) -> np.ndarray:
    """Vectorized SWAP-test Local-SDTW distance matrix.

    Replaces ~600M Python overlap_fn calls (per rank=3 combo) with a single
    einsum + one vectorized np.random.binomial draw per sequence pair.
    Expected speedup: ~50-100x over the scalar estimator path.

    Normalization matches scalar path: sum(est_overlaps_sq over rank^2) / rank.
    Negative estimates from shot noise are preserved before summing; the final
    cost is clipped at 0 (same as the scalar max(0, 1-affinity) formula).
    """
    n_test = len(test_bases_list)
    n_train = len(train_bases_list)
    distances = np.empty((n_test, n_train), dtype=np.float64)
    for i, test_bases in enumerate(test_bases_list):
        for j, train_bases in enumerate(train_bases_list):
            # True squared overlaps: (M1, M2, rank, rank)
            overlaps_sq = np.einsum("mda,ndb->mnab", test_bases, train_bases) ** 2
            # SWAP-test: P(measure |0>) = (1 + |<u|v>|^2) / 2
            p_zero = np.clip((1.0 + overlaps_sq) / 2.0, 0.0, 1.0)
            counts_zero = rng.binomial(shots, p_zero)
            # Estimated squared overlap (can be negative due to shot noise)
            est_sq = 2.0 * counts_zero / shots - 1.0
            # Affinity: sum over rank^2 pairs / rank (matches scalar normalization)
            affinity = est_sq.sum(axis=(-2, -1)) / rank  # (M1, M2)
            cost_matrix = np.clip(1.0 - affinity, 0.0, None)
            distances[i, j] = dtw_distance_from_cost_matrix(
                cost_matrix,
                normalize_by_path_length=True,
                window_ratio=dtw_window,
            )
    return distances


def _parse_window_ratio(value: str | None) -> float | None:
    if value is None or value.lower() in ("none", ""):
        return None
    return float(value)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msr_action3d"])
    parser.add_argument(
        "--backend",
        required=True,
        choices=list(BACKENDS),
        help="'exact' for deterministic Local-SDTW; 'swap' for SWAP-test quantum-estimated.",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=[10],
        help="Local window sizes in frames (default: 10).",
    )
    parser.add_argument(
        "--strides",
        nargs="+",
        type=int,
        default=[5],
        help="Window stride in frames (default: 5).",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[2],
        help="Subspace ranks per window (default: 2).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=None,
        metavar="MODE",
        help=(
            "Feature mode(s) to evaluate. "
            "One or more of: position velocity acceleration "
            "position_velocity bone_vectors bone_velocity. "
            "Default: position."
        ),
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=None,
        help="Shot counts for the SWAP backend (e.g. 128 512 2048).",
    )
    parser.add_argument(
        "--dtw-window",
        default=None,
        help="Sakoe-Chiba window ratio for the DTW alignment step (e.g. '0.1'). "
             "Default: unconstrained.",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--experiment-config", default="configs/experiment_main.yaml")
    parser.add_argument("--dataset-config", default="configs/msr_action3d.yaml")
    parser.add_argument(
        "--output",
        default="results/raw/local_sdtw_exact_phase1.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
