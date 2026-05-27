#!/usr/bin/env python
"""Aggregate raw result CSVs into a summary table with mean ± std and Wilcoxon p-values.

Usage
-----
    python scripts/aggregate_results.py results/raw/dtw_baselines.csv \\
        --output results/tables/dtw_baselines_summary_v2.csv \\
        --baseline raw_dtw \\
        --group-by method feature_mode parameters_hash

The script reads one or more raw CSV files, groups rows by the columns named in
``--group-by``, computes per-group statistics, and writes a summary CSV.  For
each non-baseline group it also computes a paired Wilcoxon p-value against the
baseline group (matched by seed).

Output columns
--------------
All ``--group-by`` columns, plus:
  acc_mean, acc_std, f1_mean, f1_std, n_seeds, runtime_mean, wilcoxon_p_vs_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.eval.aggregate import add_feature_mode, aggregate  # noqa: F401 (re-exported)
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent_dir

logger = get_logger("aggregate_results")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="RAW_CSV",
        help="One or more raw result CSV files.",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="SUMMARY_CSV",
        help="Output summary CSV path.",
    )
    p.add_argument(
        "--baseline",
        default=None,
        metavar="METHOD_NAME",
        help=(
            "Method name to use as the Wilcoxon baseline "
            "(e.g. 'raw_dtw').  Omit to skip p-value computation."
        ),
    )
    p.add_argument(
        "--group-by",
        nargs="+",
        default=["method", "feature_mode", "parameters_hash"],
        metavar="COL",
        help=(
            "Columns to group by (default: method feature_mode parameters_hash). "
            "Use 'parameters_hash' to auto-generate a hash from non-metric columns."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dfs = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            logger.error("Input file not found: %s", p)
            sys.exit(1)
        logger.info("Reading %s", p)
        dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, ignore_index=True)
    df = add_feature_mode(df)

    logger.info(
        "Loaded %d rows from %d file(s).  Group-by: %s",
        len(df),
        len(dfs),
        args.group_by,
    )

    summary = aggregate(df, group_by=args.group_by, baseline_method=args.baseline)

    output_path = ensure_parent_dir(Path(args.output))
    summary.to_csv(output_path, index=False)
    logger.info("Wrote %d summary rows to %s", len(summary), output_path)


if __name__ == "__main__":
    main()
