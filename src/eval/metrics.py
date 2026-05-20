"""Evaluation metrics used by experiment scripts."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return classification accuracy."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if true.size == 0:
        raise ValueError("Cannot compute accuracy on an empty target array.")
    return float(np.mean(true == pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return unweighted mean F1 across labels present in y_true or y_pred."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    labels = np.union1d(true, pred)
    if labels.size == 0:
        raise ValueError("Cannot compute macro-F1 on an empty target array.")

    f1_values: list[float] = []
    for label in labels:
        tp = np.sum((true == label) & (pred == label))
        fp = np.sum((true != label) & (pred == label))
        fn = np.sum((true == label) & (pred != label))
        denominator = (2 * tp) + fp + fn
        f1_values.append(float((2 * tp) / denominator) if denominator else 0.0)
    return float(np.mean(f1_values))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and a count matrix with rows=true, columns=predicted."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    labels = np.union1d(true, pred)
    index = {label: i for i, label in enumerate(labels.tolist())}
    matrix = np.zeros((labels.size, labels.size), dtype=int)
    for target, prediction in zip(true, pred):
        matrix[index[target], index[prediction]] += 1
    return labels, matrix
