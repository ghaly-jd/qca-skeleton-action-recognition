#!/usr/bin/env python
"""Generate aggregate result tables from experiment CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.utils.io import write_csv_rows
from src.utils.paths import project_path


DTW_RESULT_FIELDNAMES = [
    "dataset",
    "seed",
    "method",
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

DTW_SUMMARY_FIELDNAMES = [
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

MAIN_RESULTS_FIELDNAMES = [
    "dataset",
    "method",
    "configuration",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "runtime_mean",
    "source",
]


def main() -> None:
    args = parse_args()
    if args.table == "dtw_baselines":
        write_dtw_baseline_tables()
    elif args.table == "main_results":
        write_main_results_table()
    elif args.table == "all":
        write_dtw_baseline_tables()
        write_main_results_table()
    else:
        raise ValueError(f"Unknown table: {args.table}")


def write_dtw_baseline_tables() -> None:
    raw_rows = _read_optional_csv(project_path("results/raw/dtw_baselines.csv"))
    pca_rows = _read_many_csv(
        [
            project_path("results/raw/pca_dtw_gpu_k_04_08.csv"),
            project_path("results/raw/pca_dtw_gpu_k_12_16.csv"),
            project_path("results/raw/pca_dtw_gpu_k_24_32.csv"),
        ]
    )

    merged_rows = [_normalize_dtw_row(row) for row in raw_rows + pca_rows]
    merged_rows = _deduplicate_dtw_rows(merged_rows)
    merged_rows.sort(key=_dtw_result_sort_key)

    write_csv_rows(
        project_path("results/raw/dtw_baselines.csv"),
        merged_rows,
        fieldnames=DTW_RESULT_FIELDNAMES,
    )
    summary_rows = summarize_dtw_rows(merged_rows)
    write_csv_rows(
        project_path("results/tables/dtw_baselines_summary.csv"),
        summary_rows,
        fieldnames=DTW_SUMMARY_FIELDNAMES,
    )
    pca_only = [row for row in merged_rows if row["method"] == "pca_dtw"]
    write_csv_rows(
        project_path("results/raw/pca_dtw_baselines.csv"),
        pca_only,
        fieldnames=DTW_RESULT_FIELDNAMES,
    )
    pca_summary = [row for row in summary_rows if row["method"] == "pca_dtw"]
    write_csv_rows(
        project_path("results/tables/pca_dtw_baselines_summary.csv"),
        pca_summary,
        fieldnames=DTW_SUMMARY_FIELDNAMES,
    )


def write_main_results_table() -> None:
    dtw_summary = _read_csv(project_path("results/tables/dtw_baselines_summary.csv"))
    canonical_summary = _read_csv(
        project_path("results/tables/canonical_angles_exact_summary.csv")
    )

    rows: list[dict[str, Any]] = []
    rows.append(
        _main_result_from_dtw(
            _best_row([row for row in dtw_summary if row["method"] == "raw_dtw"]),
            method_label="Raw DTW",
        )
    )
    rows.append(
        _main_result_from_dtw(
            _best_row([row for row in dtw_summary if row["method"] == "pca_dtw"]),
            method_label="PCA+DTW",
        )
    )
    rows.append(
        _main_result_from_canonical(
            _best_row(canonical_summary),
            method_label="Exact canonical angles",
        )
    )

    write_csv_rows(
        project_path("results/tables/main_results.csv"),
        rows,
        fieldnames=MAIN_RESULTS_FIELDNAMES,
    )


def summarize_dtw_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                row["method"],
                row["pca_k"],
                row["backend"],
                row["device"],
                row["dtype"],
                row["window_ratio"],
                row["normalize_by_path_length"],
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=_dtw_summary_sort_key):
        accuracies = np.asarray([float(row["accuracy"]) for row in group_rows])
        macro_f1_values = np.asarray([float(row["macro_f1"]) for row in group_rows])
        runtimes = np.asarray([float(row["runtime_sec"]) for row in group_rows])
        summary_rows.append(
            {
                "dataset": key[0],
                "method": key[1],
                "pca_k": key[2],
                "backend": key[3],
                "device": key[4],
                "dtype": key[5],
                "window_ratio": key[6],
                "normalize_by_path_length": key[7],
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
    parser.add_argument(
        "--table",
        default="all",
        choices=["all", "dtw_baselines", "main_results"],
    )
    return parser.parse_args()


def _read_many_csv(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_optional_csv(path))
    return rows


def _read_optional_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _read_csv(path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_dtw_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field, "") for field in DTW_RESULT_FIELDNAMES}
    if normalized["method"] == "raw_dtw":
        normalized["pca_k"] = ""
    return normalized


def _deduplicate_dtw_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["seed"],
            row["method"],
            row["pca_k"],
            row["window_ratio"],
            row["normalize_by_path_length"],
        )
        unique[key] = row
    return list(unique.values())


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select best row from an empty table.")
    return max(
        rows,
        key=lambda row: (
            float(row["accuracy_mean"]),
            float(row["macro_f1_mean"]),
            -float(row["runtime_mean"]),
        ),
    )


def _main_result_from_dtw(row: dict[str, Any], *, method_label: str) -> dict[str, Any]:
    if row["method"] == "pca_dtw":
        configuration = f"k={row['pca_k']}, window={row['window_ratio']}"
    else:
        configuration = f"window={row['window_ratio']}"
    return {
        "dataset": row["dataset"],
        "method": method_label,
        "configuration": configuration,
        "accuracy_mean": row["accuracy_mean"],
        "accuracy_std": row["accuracy_std"],
        "macro_f1_mean": row["macro_f1_mean"],
        "macro_f1_std": row["macro_f1_std"],
        "runtime_mean": row["runtime_mean"],
        "source": "results/tables/dtw_baselines_summary.csv",
    }


def _main_result_from_canonical(
    row: dict[str, Any],
    *,
    method_label: str,
) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "method": method_label,
        "configuration": f"distance={row['distance']}, r={row['r']}",
        "accuracy_mean": row["accuracy_mean"],
        "accuracy_std": row["accuracy_std"],
        "macro_f1_mean": row["macro_f1_mean"],
        "macro_f1_std": row["macro_f1_std"],
        "runtime_mean": row["runtime_mean"],
        "source": "results/tables/canonical_angles_exact_summary.csv",
    }


def _dtw_result_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        _method_order(row["method"]),
        _pca_k_order(row["pca_k"]),
        _window_order(row["window_ratio"]),
        int(row["seed"]),
    )


def _dtw_summary_sort_key(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[Any, ...]:
    key = item[0]
    return (
        key[0],
        _method_order(key[1]),
        _pca_k_order(key[2]),
        _window_order(key[6]),
    )


def _method_order(method: str) -> int:
    return {"raw_dtw": 0, "pca_dtw": 1}.get(method, 99)


def _pca_k_order(value: Any) -> int:
    return -1 if value in {"", None} else int(value)


def _window_order(value: Any) -> float:
    return -1.0 if str(value) == "none" else float(value)


if __name__ == "__main__":
    main()
