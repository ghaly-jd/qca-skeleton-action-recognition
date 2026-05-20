"""Logging helpers for scripts."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Create a simple stderr logger without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    return logger

