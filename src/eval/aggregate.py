"""Core aggregation logic for raw experiment CSVs.

This module is the importable heart of ``scripts/aggregate_results.py``.
Move the heavy logic here so tests can import it without putting
``scripts/`` on ``sys.path``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Sequence

import pandas as pd

from src.eval.statistics import mean_std, paired_wilcoxon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parameters_hash(row: pd.Series) -> str:
    """Stable 12-char SHA-256 hash of the experimental-parameter columns."""
    EXCLUDE = {"accuracy", "macro_f1", "runtime_sec", "git_commit", "timestamp", "seed"}
    params = {
        k: str(v)
        for k, v in sorted(row.items())
        if k not in EXCLUDE and not pd.isna(v)
    }
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def add_parameters_hash(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a ``parameters_hash`` column appended."""
    df = df.copy()
    df["parameters_hash"] = df.apply(_parameters_hash, axis=1)
    return df


def add_feature_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``feature_mode`` filled in if missing."""
    if "feature_mode" not in df.columns:
        df = df.copy()
        df["feature_mode"] = "position"
    return df


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------


def aggregate(
    df: pd.DataFrame,
    group_by: list[str],
    baseline_method: str | None,
) -> pd.DataFrame:
    """Aggregate *df* into a summary table with statistics and Wilcoxon p-values.

    Parameters
    ----------
    df:
        Raw results table.  Must contain ``accuracy``, ``macro_f1``, and ``seed``
        columns.
    group_by:
        Column names to group by.  Each unique combination produces one output row.
        Use ``"parameters_hash"`` to auto-generate a hash from non-metric columns.
    baseline_method:
        Value in the ``method`` column used as the Wilcoxon baseline.  Pass
        ``None`` to skip p-value computation.

    Returns
    -------
    pd.DataFrame
        Summary with columns from *group_by* plus:
        ``acc_mean``, ``acc_std``, ``f1_mean``, ``f1_std``,
        ``n_seeds``, ``runtime_mean``, ``wilcoxon_p_vs_baseline``.
    """
    required = {"accuracy", "macro_f1", "seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    df = df.copy()
    for col in group_by:
        if col not in df.columns:
            if col == "feature_mode":
                df[col] = "position"
            elif col == "parameters_hash":
                df = add_parameters_hash(df)
            else:
                raise ValueError(
                    f"Group-by column '{col}' is not present in the input data."
                )

    records: list[tuple[dict, pd.DataFrame]] = []
    for keys, grp in df.groupby(group_by, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: dict = dict(zip(group_by, keys))
        accs = grp["accuracy"].dropna().tolist()
        f1s = grp["macro_f1"].dropna().tolist()
        runtimes = (
            grp["runtime_sec"].dropna().tolist() if "runtime_sec" in grp.columns else []
        )

        if accs:
            acc_mean, acc_std = mean_std(accs) if len(accs) > 1 else (accs[0], float("nan"))
        else:
            acc_mean = acc_std = float("nan")

        if f1s:
            f1_mean, f1_std = mean_std(f1s) if len(f1s) > 1 else (f1s[0], float("nan"))
        else:
            f1_mean = f1_std = float("nan")

        row["acc_mean"] = acc_mean
        row["acc_std"] = acc_std
        row["f1_mean"] = f1_mean
        row["f1_std"] = f1_std
        row["n_seeds"] = int(grp["seed"].nunique())
        row["runtime_mean"] = (
            float(sum(runtimes) / len(runtimes)) if runtimes else float("nan")
        )
        row["wilcoxon_p_vs_baseline"] = float("nan")
        records.append((row, grp))

    summary_rows = [r for r, _ in records]
    summary_df = pd.DataFrame(summary_rows)

    # Compute Wilcoxon p-values.
    if baseline_method is not None and "method" in group_by and len(records) > 1:
        if (summary_df["method"] == baseline_method).sum() == 0:
            logger.warning(
                "Baseline method '%s' not found in grouped results; "
                "skipping Wilcoxon computation.",
                baseline_method,
            )
        else:
            non_method_keys = [c for c in group_by if c != "method"]
            group_keys = [tuple(r[c] for c in group_by) for r, _ in records]

            baseline_groups: dict[tuple, pd.DataFrame] = {
                gkeys: grp
                for (row, grp), gkeys in zip(records, group_keys)
                if row.get("method") == baseline_method
            }

            for i, ((row, grp), gkeys) in enumerate(zip(records, group_keys)):
                if row.get("method") == baseline_method:
                    continue

                # Find baseline group with matching non-method keys.
                matching_key: tuple | None = None
                for bkey in baseline_groups:
                    bkey_dict = dict(zip(group_by, bkey))
                    if all(bkey_dict.get(c) == row.get(c) for c in non_method_keys):
                        matching_key = bkey
                        break

                if matching_key is None:
                    continue

                baseline_grp = baseline_groups[matching_key]
                merged = grp[["seed", "accuracy"]].merge(
                    baseline_grp[["seed", "accuracy"]],
                    on="seed",
                    suffixes=("_method", "_baseline"),
                )
                if len(merged) < 2:
                    continue

                try:
                    result = paired_wilcoxon(
                        merged["accuracy_method"].tolist(),
                        merged["accuracy_baseline"].tolist(),
                    )
                    summary_df.at[i, "wilcoxon_p_vs_baseline"] = result["p_value"]
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Wilcoxon failed: %s", exc)

    return summary_df
