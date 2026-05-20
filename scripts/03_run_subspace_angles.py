#!/usr/bin/env python
"""Run exact canonical-angle 1-NN classification experiments."""

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
from src.distances.canonical_angles import pairwise_canonical_singular_values
from src.distances.subspace_distances import DISTANCE_NAMES, distance_from_singular_values
from src.eval.knn import predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import get_git_commit, utc_timestamp
from src.features.sequence_subspace import compute_subspaces, stack_bases
from src.utils.io import read_json, read_yaml, write_csv_rows
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent_dir, project_path


RESULT_FIELDNAMES = [
    "dataset",
    "seed",
    "r",
    "distance",
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
    "distance",
    "r",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "runtime_mean",
]


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is implemented for Phase 2.")

    logger = get_logger("subspace_angles")
    experiment_config = read_yaml(project_path(args.experiment_config))
    dataset_config = read_yaml(project_path(args.dataset_config))

    r_values = args.r_values or experiment_config["subspace"]["r_values"]
    distances = args.distance or experiment_config["canonical_angles"]["distances"]
    seeds = args.seeds or experiment_config["seeds"]
    _validate_distances(distances)

    processed_dir = _resolve_path(dataset_config["dataset"]["processed_dir"])
    dataset = load_processed_dataset(processed_dir)
    rows = run_experiment(
        dataset,
        seeds=[int(seed) for seed in seeds],
        r_values=[int(rank) for rank in r_values],
        distances=distances,
        min_frames_required=int(experiment_config["subspace"]["min_frames_required"]),
        center_sequence=bool(experiment_config["subspace"]["center_sequence"]),
        feature_standardization=args.feature_standardization,
        frame_l2_normalization=args.l2_normalize_frames,
        logger=logger,
    )

    output_path = _resolve_path(args.output)
    write_csv_rows(output_path, rows, fieldnames=RESULT_FIELDNAMES)
    logger.info("Wrote %s rows to %s.", len(rows), output_path)

    summary_rows = summarize_rows(rows)
    summary_path = _resolve_path(args.summary_output)
    write_csv_rows(summary_path, summary_rows, fieldnames=SUMMARY_FIELDNAMES)
    logger.info("Wrote summary table to %s.", summary_path)

    if not args.no_plot:
        plot_path = _resolve_path(args.plot_output)
        write_accuracy_vs_rank_plot(summary_rows, plot_path)
        logger.info("Wrote rank sweep plot to %s.", plot_path)

    best = max(
        (row for row in rows if row["distance"] == "chordal"),
        key=lambda row: float(row["accuracy"]),
    )
    logger.info(
        "Best chordal row: seed=%s r=%s accuracy=%.4f macro_f1=%.4f",
        best["seed"],
        best["r"],
        float(best["accuracy"]),
        float(best["macro_f1"]),
    )


def run_experiment(
    dataset: ProcessedDataset,
    *,
    seeds: list[int],
    r_values: list[int],
    distances: list[str],
    min_frames_required: int,
    center_sequence: bool,
    feature_standardization: str,
    frame_l2_normalization: bool,
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
            if not train_indices or not test_indices:
                raise ValueError(
                    f"Empty train/test split after frame filtering for rank={rank}."
                )

            extract_start = perf_counter()
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
            y_train = dataset.labels[train_indices]
            y_test = dataset.labels[test_indices]
            extraction_runtime = perf_counter() - extract_start

            singular_start = perf_counter()
            singular_values = pairwise_canonical_singular_values(test_bases, train_bases)
            singular_runtime = perf_counter() - singular_start

            for distance in distances:
                classify_start = perf_counter()
                distance_matrix = distance_from_singular_values(
                    singular_values,
                    metric=distance,
                )
                result = predict_1nn_from_distances(distance_matrix, y_train)
                classification_runtime = perf_counter() - classify_start
                runtime = extraction_runtime + singular_runtime + classification_runtime

                row = {
                    "dataset": dataset.dataset,
                    "seed": seed,
                    "r": rank,
                    "distance": distance,
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
                    "seed=%s r=%s distance=%s accuracy=%.4f macro_f1=%.4f",
                    seed,
                    rank,
                    distance,
                    row["accuracy"],
                    row["macro_f1"],
                )

    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["distance"]), int(row["r"]))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, distance, rank), group_rows in sorted(groups.items()):
        accuracies = np.asarray([float(row["accuracy"]) for row in group_rows])
        macro_f1_values = np.asarray([float(row["macro_f1"]) for row in group_rows])
        runtimes = np.asarray([float(row["runtime_sec"]) for row in group_rows])
        summary_rows.append(
            {
                "dataset": dataset,
                "distance": distance,
                "r": rank,
                "accuracy_mean": round(float(accuracies.mean()), 6),
                "accuracy_std": round(float(accuracies.std(ddof=0)), 6),
                "macro_f1_mean": round(float(macro_f1_values.mean()), 6),
                "macro_f1_std": round(float(macro_f1_values.std(ddof=0)), 6),
                "runtime_mean": round(float(runtimes.mean()), 6),
            }
        )
    return summary_rows


def write_accuracy_vs_rank_plot(summary_rows: list[dict[str, Any]], path: Path) -> None:
    """Write the initial accuracy-vs-rank figure."""
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qca_skeleton_matplotlib")
    import matplotlib.pyplot as plt

    path = ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    distances = sorted({str(row["distance"]) for row in summary_rows})
    for distance in distances:
        rows = [row for row in summary_rows if row["distance"] == distance]
        rows.sort(key=lambda row: int(row["r"]))
        x = [int(row["r"]) for row in rows]
        y = [float(row["accuracy_mean"]) for row in rows]
        yerr = [float(row["accuracy_std"]) for row in rows]
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.6, capsize=3, label=distance)

    ax.set_xlabel("Subspace rank r")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Distance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msr_action3d"])
    parser.add_argument("--experiment-config", default="configs/experiment_main.yaml")
    parser.add_argument("--dataset-config", default="configs/msr_action3d.yaml")
    parser.add_argument("--r-values", nargs="+", type=int, default=None)
    parser.add_argument("--distance", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--feature-standardization", default="none")
    parser.add_argument("--l2-normalize-frames", action="store_true")
    parser.add_argument("--output", default="results/raw/canonical_angles_exact.csv")
    parser.add_argument(
        "--summary-output",
        default="results/tables/canonical_angles_exact_summary.csv",
    )
    parser.add_argument("--plot-output", default="results/figures/accuracy_vs_rank.png")
    parser.add_argument("--no-plot", action="store_true")
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


def _validate_distances(distances: list[str]) -> None:
    unknown = sorted(set(distances) - set(DISTANCE_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown distances {unknown}. Known values: {', '.join(DISTANCE_NAMES)}."
        )


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


if __name__ == "__main__":
    main()
