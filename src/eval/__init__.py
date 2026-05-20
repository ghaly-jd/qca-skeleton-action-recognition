"""Evaluation, statistics, metrics, and result-writing utilities."""
"""Evaluation helpers."""

from src.eval.knn import KNNResult, classify_1nn_subspaces, predict_1nn_from_distances
from src.eval.metrics import accuracy, macro_f1

__all__ = [
    "KNNResult",
    "accuracy",
    "classify_1nn_subspaces",
    "macro_f1",
    "predict_1nn_from_distances",
]
