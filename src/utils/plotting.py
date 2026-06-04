"""Shared matplotlib style for paper figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent color map for feature modes
FEATURE_COLORS = {
    "position":          "#4878CF",
    "velocity":          "#6ACC65",
    "position_velocity": "#D65F5F",
    "bone_vectors":      "#B47CC7",
    "bone_velocity":     "#C4AD66",
}

FEATURE_LABELS = {
    "position":          "Position",
    "velocity":          "Velocity",
    "position_velocity": "Pos+Vel",
    "bone_vectors":      "Bone Vecs",
    "bone_velocity":     "Bone+Vel",
}

METHOD_DISPLAY = {
    "local_sdtw_exact": "Exact\nLocal-SDTW",
    "local_sdtw_swap":  "SWAP\nLocal-SDTW",
    "raw_dtw":          "Raw DTW",
    "pca_dtw":          "PCA+DTW",
    "kdtw":             "KDTW",
    "gak":              "GAK",
    "mlp":              "MLP",
    "random_forest":    "Rand.\nForest",
    "lstm":             "LSTM",
}

FIGURE_DPI = 150
FIGURE_FORMAT = ["png", "pdf"]


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   9,
        "legend.fontsize":   8,
        "figure.dpi":        FIGURE_DPI,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })
