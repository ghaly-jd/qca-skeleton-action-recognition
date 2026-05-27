"""Tests for src/eval/statistics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.statistics import bootstrap_ci, mean_std, paired_wilcoxon


class TestMeanStd:
    def test_matches_numpy(self):
        values = [0.1, 0.5, 0.3, 0.8, 0.6, 0.4, 0.7, 0.2, 0.9, 0.55]
        m, s = mean_std(values)
        assert m == pytest.approx(np.mean(values))
        assert s == pytest.approx(np.std(values, ddof=1))

    def test_single_element_std_is_nan(self):
        """std with ddof=1 of a single element is NaN."""
        m, s = mean_std([0.5])
        assert m == pytest.approx(0.5)
        assert np.isnan(s)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean_std([])


class TestPairedWilcoxon:
    def test_identical_inputs_p_value_is_nan(self):
        """Identical inputs have zero differences; p-value is NaN."""
        a = [0.8, 0.7, 0.9, 0.75, 0.85, 0.6, 0.65, 0.7, 0.8, 0.9]
        result = paired_wilcoxon(a, a)
        assert np.isnan(result["p_value"])

    def test_strictly_larger_inputs_significant(self):
        """Strictly larger a → clearly significant two-sided test (n=10)."""
        a = [0.8, 0.82, 0.85, 0.9, 0.88, 0.86, 0.84, 0.87, 0.91, 0.83]
        b = [0.6, 0.62, 0.65, 0.70, 0.68, 0.66, 0.64, 0.67, 0.71, 0.63]
        result = paired_wilcoxon(a, b)
        assert result["p_value"] < 0.05

    def test_n_pairs_returned(self):
        a = [0.5] * 8
        b = [0.4] * 8
        result = paired_wilcoxon(a, b)
        assert result["n_pairs"] == 8

    def test_alternative_is_two_sided(self):
        a = [0.6, 0.7, 0.8, 0.9]
        b = [0.5, 0.6, 0.7, 0.8]
        result = paired_wilcoxon(a, b)
        assert result["alternative"] == "two-sided"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            paired_wilcoxon([0.5, 0.6], [0.5])

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError):
            paired_wilcoxon([0.5], [0.6])

    def test_returns_dict_keys(self):
        a = [0.7, 0.8, 0.9, 0.75, 0.85]
        b = [0.6, 0.7, 0.8, 0.65, 0.75]
        result = paired_wilcoxon(a, b)
        assert set(result.keys()) == {"statistic", "p_value", "n_pairs", "alternative"}


class TestBootstrapCI:
    def test_lower_lt_upper(self):
        values = [0.8, 0.82, 0.85, 0.9, 0.88, 0.86, 0.84, 0.87, 0.91, 0.83]
        lo, hi = bootstrap_ci(values)
        assert lo < hi

    def test_mean_within_ci(self):
        values = [0.8, 0.82, 0.85, 0.9, 0.88, 0.86, 0.84, 0.87, 0.91, 0.83]
        lo, hi = bootstrap_ci(values, n_resamples=5_000)
        m = float(np.mean(values))
        assert lo <= m <= hi

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])
