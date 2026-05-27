"""Statistical helper functions for experiment evaluation.

Functions
---------
mean_std(values)
    Return (mean, std) with ddof=1 (sample standard deviation).

paired_wilcoxon(values_a, values_b)
    Two-sided paired Wilcoxon signed-rank test via ``scipy.stats.wilcoxon``.

bootstrap_ci(values, n_resamples, ci)
    Bootstrap percentile confidence interval.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon as _scipy_wilcoxon


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` with sample standard deviation (ddof=1).

    Parameters
    ----------
    values:
        A non-empty sequence of numeric values.

    Returns
    -------
    tuple[float, float]
        ``(mean, std)`` where ``std`` uses ``ddof=1``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty.")
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def paired_wilcoxon(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> dict[str, object]:
    """Compute a two-sided paired Wilcoxon signed-rank test.

    Parameters
    ----------
    values_a, values_b:
        Paired sequences of the same length (≥ 2 non-zero-difference pairs
        required; otherwise ``p_value`` is returned as ``float('nan')``).

    Returns
    -------
    dict with keys:
        - ``"statistic"`` — Wilcoxon test statistic (float).
        - ``"p_value"``   — two-sided p-value (float; NaN if test cannot be run).
        - ``"n_pairs"``   — number of pairs (int).
        - ``"alternative"`` — always ``"two-sided"``.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("values_a and values_b must be 1-D arrays of the same length.")
    if len(a) < 2:
        raise ValueError("At least 2 pairs are required.")

    differences = a - b
    n_nonzero = int(np.sum(differences != 0))

    if n_nonzero < 1:
        # All differences are zero; test statistic is undefined.
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": len(a),
            "alternative": "two-sided",
        }

    result = _scipy_wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": len(a),
        "alternative": "two-sided",
    }


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute a bootstrap percentile confidence interval for the mean.

    Parameters
    ----------
    values:
        A sequence of numeric values.
    n_resamples:
        Number of bootstrap resamples (default: 10 000).
    ci:
        Confidence level in ``(0, 1)`` (default: 0.95 → 95 % CI).

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` bounds of the confidence interval.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty.")
    rng = np.random.default_rng(seed=0)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)]
    )
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return lower, upper
