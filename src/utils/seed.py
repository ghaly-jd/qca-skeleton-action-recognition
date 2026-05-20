"""Deterministic seed helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedState:
    """Record which seed was applied."""

    seed: int
    numpy_available: bool


def set_global_seed(seed: int) -> SeedState:
    """Seed Python, hash randomization, and NumPy when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_available = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_available = True
    except ImportError:
        numpy_available = False

    return SeedState(seed=seed, numpy_available=numpy_available)

