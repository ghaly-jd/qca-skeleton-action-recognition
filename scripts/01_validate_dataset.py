#!/usr/bin/env python
"""Validate processed skeleton datasets and write dataset summary tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.data.validation import validate_processed_dataset, write_dataset_summary
from src.utils.io import read_yaml
from src.utils.logging import get_logger
from src.utils.paths import project_path


def main() -> None:
    args = parse_args()
    if args.dataset != "msr_action3d":
        raise ValueError("Only --dataset msr_action3d is implemented in Phase 1.")

    logger = get_logger("validate_dataset")
    config = read_yaml(project_path(args.config))
    dataset_config = config["dataset"]

    processed_dir = _resolve_path(args.processed_dir or dataset_config["processed_dir"])
    split_path = _resolve_path(args.split_path)
    output_path = _resolve_path(args.output)

    summary, issues = validate_processed_dataset(
        processed_dir,
        split_path=split_path,
        expected_feature_dim=int(dataset_config["feature_dim"]),
    )
    write_dataset_summary(output_path, summary)

    logger.info("Dataset summary: %s", summary)
    logger.info("Wrote %s.", output_path)

    if issues:
        for issue in issues:
            logger.error("Validation issue: %s", issue)
        raise SystemExit(1)

    logger.info("Validation passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msr_action3d"])
    parser.add_argument("--config", default="configs/msr_action3d.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--split-path", default="data/splits/msr_cross_subject.json")
    parser.add_argument("--output", default="results/tables/dataset_summary.csv")
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(path)


if __name__ == "__main__":
    main()
