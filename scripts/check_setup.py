#!/usr/bin/env python
"""Smoke-check Phase 0 imports and result writing."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src import __version__
from src.eval.result_writer import ResultRecord, append_result
from src.utils.logging import get_logger
from src.utils.paths import PROJECT_ROOT, project_path
from src.utils.seed import set_global_seed


def main() -> None:
    logger = get_logger("check_setup")
    seed_state = set_global_seed(0)
    output_path = project_path("results", "raw", "setup_check.csv")

    append_result(
        output_path,
        ResultRecord(
            dataset="setup",
            method="phase0_smoke_check",
            feature_mode="position",
            seed=seed_state.seed,
            parameters={
                "project_root": str(PROJECT_ROOT),
                "package_version": __version__,
                "numpy_available": seed_state.numpy_available,
            },
            accuracy=None,
            macro_f1=None,
            runtime_sec=0.0,
        ),
    )

    logger.info("Phase 0 setup check passed.")
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
