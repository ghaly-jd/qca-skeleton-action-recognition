#!/usr/bin/env python
"""Generate figures for Q-SDTW experiments."""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
warnings.filterwarnings(
    "ignore",
    message="Pandas requires version .*",
    category=UserWarning,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from src.utils.paths import ensure_parent_dir, project_path


def main() -> None:
    args = parse_args()
    if args.figure in {"all", "local_sdtw_heatmaps"}:
        write_local_sdtw_heatmaps(args)
    if args.figure == "all":
        if _resolve_path(args.swap_global_summary_input).exists():
            write_swap_global_convergence_figures(args)
        if _resolve_path(args.swap_local_summary_input).exists():
            write_swap_local_sdtw_convergence_figures(args)
    elif args.figure == "swap_global_convergence":
        write_swap_global_convergence_figures(args)
    elif args.figure == "swap_local_sdtw_convergence":
        write_swap_local_sdtw_convergence_figures(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        default="all",
        choices=[
            "all",
            "local_sdtw_heatmaps",
            "swap_global_convergence",
            "swap_local_sdtw_convergence",
        ],
    )
    parser.add_argument(
        "--summary-input",
        default="results/tables/local_subspace_dtw_summary_msr_action3d.csv",
    )
    parser.add_argument(
        "--accuracy-output",
        default="results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png",
    )
    parser.add_argument(
        "--runtime-output",
        default="results/figures/local_sdtw_runtime_heatmap_msr_action3d.png",
    )
    parser.add_argument("--feature-mode", default="position")
    parser.add_argument("--local-distance", default="projection_affinity")
    parser.add_argument("--dtw-window", default="none")
    parser.add_argument(
        "--swap-global-summary-input",
        default="results/tables/swap_global_exact_comparison_summary_msr_action3d.csv",
    )
    parser.add_argument(
        "--swap-global-mae-output",
        default="results/figures/swap_global_mae_vs_shots_msr_action3d.png",
    )
    parser.add_argument(
        "--swap-global-accuracy-output",
        default="results/figures/swap_global_accuracy_vs_shots_msr_action3d.png",
    )
    parser.add_argument(
        "--swap-global-agreement-output",
        default="results/figures/swap_global_prediction_agreement_msr_action3d.png",
    )
    parser.add_argument(
        "--swap-local-summary-input",
        default="results/tables/swap_local_sdtw_full_summary_msr_action3d.csv",
    )
    parser.add_argument(
        "--swap-local-mae-output",
        default="results/figures/swap_local_sdtw_mae_vs_shots_msr_action3d.png",
    )
    parser.add_argument(
        "--swap-local-accuracy-output",
        default="results/figures/swap_local_sdtw_accuracy_vs_shots_msr_action3d.png",
    )
    parser.add_argument(
        "--swap-local-agreement-output",
        default="results/figures/swap_local_sdtw_prediction_agreement_msr_action3d.png",
    )
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args()


def write_local_sdtw_heatmaps(args: argparse.Namespace) -> None:
    summary_path = _resolve_path(args.summary_input)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Local-SDTW summary table does not exist: {summary_path}"
        )

    summary = pd.read_csv(summary_path)
    filtered = _filter_local_sdtw_summary(
        summary,
        feature_mode=args.feature_mode,
        local_distance=args.local_distance,
        dtw_window=args.dtw_window,
    )
    if filtered.empty:
        raise ValueError(
            "No Local-SDTW rows match "
            f"feature_mode={args.feature_mode}, "
            f"local_distance={args.local_distance}, dtw_window={args.dtw_window}."
        )

    _write_faceted_heatmap(
        filtered,
        facet_column="stride",
        index_column="r",
        value_column="accuracy_mean",
        title="Local Subspace-DTW accuracy",
        value_format=".3f",
        output_path=_resolve_path(args.accuracy_output),
        cmap=args.cmap,
    )
    _write_faceted_heatmap(
        filtered,
        facet_column="r",
        index_column="stride",
        value_column="runtime_mean",
        title="Local Subspace-DTW runtime (s)",
        value_format=".1f",
        output_path=_resolve_path(args.runtime_output),
        cmap=args.cmap,
    )


def write_swap_global_convergence_figures(args: argparse.Namespace) -> None:
    summary_path = _resolve_path(args.swap_global_summary_input)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"SWAP global convergence summary table does not exist: {summary_path}"
        )

    summary = pd.read_csv(summary_path)
    required_columns = {
        "r",
        "shots",
        "exact_accuracy_mean",
        "quantum_accuracy_mean",
        "mean_abs_distance_error_mean",
        "nearest_neighbor_agreement_mean",
        "prediction_agreement_mean",
    }
    missing = sorted(required_columns - set(summary.columns))
    if missing:
        raise ValueError(
            f"SWAP global summary table is missing required columns: {missing}"
        )

    data = summary.copy()
    for column in required_columns:
        data[column] = pd.to_numeric(data[column])
    data = data.sort_values(["r", "shots"])

    _write_rank_lineplot(
        data,
        y_column="mean_abs_distance_error_mean",
        ylabel="Distance MAE",
        title="SWAP global distance error vs shots",
        output_path=_resolve_path(args.swap_global_mae_output),
    )
    _write_swap_accuracy_plot(
        data,
        title="SWAP global accuracy vs shots",
        output_path=_resolve_path(args.swap_global_accuracy_output),
    )
    _write_swap_agreement_plot(
        data,
        title="SWAP global agreement vs shots",
        output_path=_resolve_path(args.swap_global_agreement_output),
    )


def write_swap_local_sdtw_convergence_figures(args: argparse.Namespace) -> None:
    summary_path = _resolve_path(args.swap_local_summary_input)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"SWAP Local-SDTW summary table does not exist: {summary_path}"
        )

    summary = pd.read_csv(summary_path)
    required_columns = {
        "r",
        "shots",
        "exact_accuracy_mean",
        "quantum_accuracy_mean",
        "mean_abs_distance_error_mean",
        "nearest_neighbor_agreement_mean",
        "prediction_agreement_mean",
    }
    missing = sorted(required_columns - set(summary.columns))
    if missing:
        raise ValueError(
            f"SWAP Local-SDTW summary table is missing required columns: {missing}"
        )

    data = summary.copy()
    for column in required_columns:
        data[column] = pd.to_numeric(data[column])
    data = data.sort_values(["r", "shots"])

    _write_rank_lineplot(
        data,
        y_column="mean_abs_distance_error_mean",
        ylabel="DTW distance MAE",
        title="SWAP Local-SDTW distance error vs shots",
        output_path=_resolve_path(args.swap_local_mae_output),
    )
    _write_swap_accuracy_plot(
        data,
        title="SWAP Local-SDTW accuracy vs shots",
        output_path=_resolve_path(args.swap_local_accuracy_output),
    )
    _write_swap_agreement_plot(
        data,
        title="SWAP Local-SDTW agreement vs shots",
        output_path=_resolve_path(args.swap_local_agreement_output),
    )


def _filter_local_sdtw_summary(
    summary: pd.DataFrame,
    *,
    feature_mode: str,
    local_distance: str,
    dtw_window: str,
) -> pd.DataFrame:
    required_columns = {
        "feature_mode",
        "window_length",
        "stride",
        "r",
        "local_distance",
        "dtw_window",
        "accuracy_mean",
        "runtime_mean",
    }
    missing = sorted(required_columns - set(summary.columns))
    if missing:
        raise ValueError(f"Summary table is missing required columns: {missing}")

    filtered = summary[
        (summary["feature_mode"].astype(str) == feature_mode)
        & (summary["local_distance"].astype(str) == local_distance)
        & (summary["dtw_window"].astype(str) == dtw_window)
    ].copy()
    filtered["window_length"] = filtered["window_length"].astype(int)
    filtered["stride"] = filtered["stride"].astype(int)
    filtered["r"] = filtered["r"].astype(int)
    return filtered


def _write_faceted_heatmap(
    data: pd.DataFrame,
    *,
    facet_column: str,
    index_column: str,
    value_column: str,
    title: str,
    value_format: str,
    output_path: Path,
    cmap: str,
) -> None:
    facets = sorted(data[facet_column].unique())
    ncols = min(len(facets), 3)
    nrows = math.ceil(len(facets) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 3.8 * nrows),
        constrained_layout=True,
    )
    axes_list = _flatten_axes(axes)

    for axis, facet in zip(axes_list, facets):
        facet_data = data[data[facet_column] == facet]
        heatmap_data = facet_data.pivot_table(
            index=index_column,
            columns="window_length",
            values=value_column,
            aggfunc="mean",
        ).sort_index(axis=0).sort_index(axis=1)
        sns.heatmap(
            heatmap_data,
            ax=axis,
            annot=True,
            fmt=value_format,
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar=True,
        )
        axis.set_title(f"{facet_column}={facet}")
        axis.set_xlabel("window_length")
        axis.set_ylabel(index_column)

    for axis in axes_list[len(facets) :]:
        axis.axis("off")

    fig.suptitle(title)
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_rank_lineplot(
    data: pd.DataFrame,
    *,
    y_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for rank, rank_data in data.groupby("r", sort=True):
        axis.plot(
            rank_data["shots"],
            rank_data[y_column],
            marker="o",
            linewidth=2.0,
            label=f"r={int(rank)}",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("shots")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(title="rank")
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_swap_accuracy_plot(
    data: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for rank, rank_data in data.groupby("r", sort=True):
        axis.plot(
            rank_data["shots"],
            rank_data["exact_accuracy_mean"],
            linestyle="--",
            linewidth=1.8,
            label=f"exact r={int(rank)}",
        )
        axis.plot(
            rank_data["shots"],
            rank_data["quantum_accuracy_mean"],
            marker="o",
            linewidth=2.0,
            label=f"SWAP r={int(rank)}",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("shots")
    axis.set_ylabel("accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(ncols=2)
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_swap_agreement_plot(
    data: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10.5, 4.4),
        constrained_layout=True,
        sharey=True,
    )
    panels = [
        ("nearest_neighbor_agreement_mean", "Nearest-neighbor agreement"),
        ("prediction_agreement_mean", "Prediction agreement"),
    ]
    for axis, (column, title) in zip(axes, panels):
        for rank, rank_data in data.groupby("r", sort=True):
            axis.plot(
                rank_data["shots"],
                rank_data[column],
                marker="o",
                linewidth=2.0,
                label=f"r={int(rank)}",
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("shots")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("agreement")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].legend(title="rank")
    fig.suptitle(title)
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _flatten_axes(axes: Any) -> list[Any]:
    if isinstance(axes, Axes):
        return [axes]
    return list(axes.flat)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


if __name__ == "__main__":
    main()
