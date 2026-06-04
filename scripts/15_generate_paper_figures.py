#!/usr/bin/env python
"""Generate all paper figures from aggregated result tables.

Usage
-----
    # Phase 1 full-data comparison figure
    python scripts/15_generate_paper_figures.py --figure phase1_comparison

    # All figures
    python scripts/15_generate_paper_figures.py --figure all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import ensure_parent_dir
from src.utils.plotting import (
    FEATURE_COLORS,
    FEATURE_LABELS,
    FIGURE_DPI,
    FIGURE_FORMAT,
    METHOD_DISPLAY,
    apply_paper_style,
)

logger = get_logger("generate_paper_figures")

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for fmt in FIGURE_FORMAT:
        p = ensure_parent_dir(FIGURES_DIR / f"{stem}.{fmt}")
        fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        paths.append(p)
        logger.info("Saved %s", p)
    return paths


# ---------------------------------------------------------------------------
# Figure: Phase 1 full-data comparison
# ---------------------------------------------------------------------------

# Method order (best to worst overall)
_METHOD_ORDER = [
    "local_sdtw_exact",
    "local_sdtw_swap",
    "raw_dtw",
    "pca_dtw",
    "kdtw",
    "gak",
    "mlp",
    "random_forest",
    "lstm",
]

_FEATURE_ORDER = [
    "position",
    "velocity",
    "position_velocity",
    "bone_vectors",
    "bone_velocity",
]


def figure_phase1_comparison(
    summary_path: Path = PROJECT_ROOT / "results" / "tables" / "phase1_main_summary.csv",
) -> list[Path]:
    """Grouped bar chart: x=method, colour=feature_mode, y=accuracy ± std."""
    apply_paper_style()

    df = pd.read_csv(summary_path)

    # Restrict to methods present in the summary
    methods = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    features = [f for f in _FEATURE_ORDER if f in df["feature_mode"].unique()]

    n_methods = len(methods)
    n_features = len(features)
    bar_width = 0.13
    group_gap = 0.05
    group_width = n_features * bar_width + group_gap

    fig, ax = plt.subplots(figsize=(14, 5))

    x_centers = np.arange(n_methods) * group_width
    offsets = np.linspace(
        -(n_features - 1) / 2 * bar_width,
        (n_features - 1) / 2 * bar_width,
        n_features,
    )

    for fi, feat in enumerate(features):
        accs, stds = [], []
        for meth in methods:
            row = df[(df["method"] == meth) & (df["feature_mode"] == feat)]
            if row.empty:
                accs.append(0.0)
                stds.append(0.0)
            else:
                accs.append(float(row["acc_mean"].iloc[0]))
                s = float(row["acc_std"].iloc[0])
                stds.append(0.0 if np.isnan(s) else s)

        bars = ax.bar(
            x_centers + offsets[fi],
            accs,
            width=bar_width,
            color=FEATURE_COLORS[feat],
            label=FEATURE_LABELS[feat],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        # Error bars only where std > 0
        for xi, (acc, std) in enumerate(zip(accs, stds)):
            if std > 1e-6:
                ax.errorbar(
                    x_centers[xi] + offsets[fi],
                    acc,
                    yerr=std,
                    fmt="none",
                    color="black",
                    capsize=2,
                    linewidth=0.8,
                )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [METHOD_DISPLAY.get(m, m) for m in methods],
        ha="center",
        multialignment="center",
    )
    ax.set_ylabel("Accuracy (mean over 10 seeds)")
    ax.set_title(
        "Phase 1 — Full-Data MSR Action3D: All Methods × Feature Modes",
        pad=10,
    )
    ax.set_ylim(0.45, 0.95)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Significance markers above the best feature bar per method
    for mi, meth in enumerate(methods):
        best_row = df[df["method"] == meth].sort_values("acc_mean", ascending=False)
        if best_row.empty:
            continue
        p = best_row["wilcoxon_p_vs_baseline"].iloc[0]
        acc = float(best_row["acc_mean"].iloc[0])
        feat = best_row["feature_mode"].iloc[0]
        fi = features.index(feat) if feat in features else 0
        marker = ""
        if pd.notna(p):
            if p < 0.01:
                marker = "**"
            elif p < 0.05:
                marker = "*"
        if marker:
            ax.text(
                x_centers[mi] + offsets[fi],
                acc + 0.005,
                marker,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    ax.legend(title="Feature mode", loc="lower right", framealpha=0.8)
    ax.axhline(0.8545, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(
        n_methods * group_width - group_width * 0.3,
        0.857,
        "Raw DTW best (0.855)",
        fontsize=7,
        color="gray",
    )

    fig.tight_layout()
    return _save(fig, "phase1_full_data_comparison")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIGURE_MAP = {
    "phase1_comparison": figure_phase1_comparison,
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--figure",
        choices=list(FIGURE_MAP) + ["all"],
        default="all",
        help="Which figure to generate (default: all).",
    )
    args = p.parse_args(argv)

    targets = list(FIGURE_MAP) if args.figure == "all" else [args.figure]
    for name in targets:
        logger.info("Generating figure: %s", name)
        paths = FIGURE_MAP[name]()
        for path in paths:
            print(f"  → {path}")


if __name__ == "__main__":
    main()
