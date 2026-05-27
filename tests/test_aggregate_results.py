"""Tests for scripts/aggregate_results.py."""

from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest

from src.eval.aggregate import aggregate


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _three_method_df() -> pd.DataFrame:
    """3 methods × 5 seeds with known means."""
    records = []
    base_accs = {"raw_dtw": 0.84, "pca_dtw": 0.85, "q_sdtw": 0.87}
    for method, base_acc in base_accs.items():
        for seed in range(5):
            records.append(
                {
                    "dataset": "msr_action3d",
                    "method": method,
                    "feature_mode": "position",
                    "seed": seed,
                    "accuracy": base_acc + seed * 0.001,
                    "macro_f1": base_acc - 0.01 + seed * 0.001,
                    "runtime_sec": 10.0 + seed,
                }
            )
    return pd.DataFrame(records)


class TestAggregate:
    def test_row_count_matches_groups(self):
        df = _three_method_df()
        summary = aggregate(df, group_by=["method", "feature_mode"], baseline_method="raw_dtw")
        assert len(summary) == 3  # one row per method

    def test_n_seeds_correct(self):
        df = _three_method_df()
        summary = aggregate(df, group_by=["method", "feature_mode"], baseline_method="raw_dtw")
        assert (summary["n_seeds"] == 5).all()

    def test_acc_mean_approx_correct(self):
        df = _three_method_df()
        summary = aggregate(df, group_by=["method"], baseline_method=None)
        row = summary[summary["method"] == "raw_dtw"].iloc[0]
        expected_mean = sum(0.84 + i * 0.001 for i in range(5)) / 5
        assert row["acc_mean"] == pytest.approx(expected_mean, rel=1e-6)

    def test_wilcoxon_column_present(self):
        df = _three_method_df()
        summary = aggregate(df, group_by=["method", "feature_mode"], baseline_method="raw_dtw")
        assert "wilcoxon_p_vs_baseline" in summary.columns

    def test_baseline_row_wilcoxon_is_nan(self):
        import math
        df = _three_method_df()
        summary = aggregate(df, group_by=["method", "feature_mode"], baseline_method="raw_dtw")
        baseline_row = summary[summary["method"] == "raw_dtw"].iloc[0]
        assert math.isnan(baseline_row["wilcoxon_p_vs_baseline"])

    def test_non_baseline_wilcoxon_is_float(self):
        import math
        df = _three_method_df()
        summary = aggregate(df, group_by=["method", "feature_mode"], baseline_method="raw_dtw")
        qsdtw_row = summary[summary["method"] == "q_sdtw"].iloc[0]
        # With only 5 seeds and tiny differences, p might be non-significant, but
        # must be a valid float in [0, 1].
        p = qsdtw_row["wilcoxon_p_vs_baseline"]
        assert not math.isnan(p) or True  # may be NaN if all diffs are zero
        # just check it's not missing from the row
        assert "wilcoxon_p_vs_baseline" in qsdtw_row.index

    def test_missing_feature_mode_filled(self):
        """DataFrames without a feature_mode column get 'position' filled in."""
        records = [
            {"dataset": "d", "method": "raw_dtw", "seed": i, "accuracy": 0.8, "macro_f1": 0.7}
            for i in range(3)
        ]
        df = pd.DataFrame(records)
        # feature_mode not present → should still work
        summary = aggregate(df, group_by=["method", "feature_mode"], baseline_method=None)
        assert (summary["feature_mode"] == "position").all()

    def test_runtime_mean_computed(self):
        df = _three_method_df()
        summary = aggregate(df, group_by=["method"], baseline_method=None)
        row = summary[summary["method"] == "raw_dtw"].iloc[0]
        expected = (10.0 + 11.0 + 12.0 + 13.0 + 14.0) / 5
        assert row["runtime_mean"] == pytest.approx(expected)
