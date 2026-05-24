"""Common interface for quantum overlap estimators."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from src.quantum.swap_test import estimate_squared_overlap_swap_test


@dataclass(frozen=True)
class OverlapEstimate:
    """A squared-overlap estimate with measurement metadata."""

    overlap_squared: float
    probability_zero: float
    counts_zero: int
    counts_one: int
    shots: int
    method: str
    simulator: str
    seed: int | None


class SwapTestOverlapEstimator:
    """Estimate squared overlaps through the SWAP-test measurement model."""

    def __init__(
        self,
        *,
        shots: int,
        simulator: str = "sampling",
        seed: int | None = None,
        cache: bool = True,
    ) -> None:
        if shots <= 0:
            raise ValueError("shots must be positive.")
        self.shots = int(shots)
        self.simulator = simulator
        self.seed = seed
        self.cache = bool(cache)
        self.num_estimates = 0
        self.num_cache_hits = 0
        self._cache: dict[str, OverlapEstimate] = {}
        self._rng = np.random.default_rng(seed)

    def estimate(self, vector_x: np.ndarray, vector_y: np.ndarray) -> OverlapEstimate:
        """Estimate ``|<x|y>|^2`` for two real vectors."""
        key = _cache_key(vector_x, vector_y, shots=self.shots, simulator=self.simulator)
        if self.cache and key in self._cache:
            self.num_cache_hits += 1
            return self._cache[key]

        call_seed = self._next_seed()
        result = estimate_squared_overlap_swap_test(
            vector_x,
            vector_y,
            shots=self.shots,
            simulator=self.simulator,
            seed=call_seed,
        )
        estimate = OverlapEstimate(
            overlap_squared=result.overlap_squared,
            probability_zero=result.probability_zero,
            counts_zero=result.counts_zero,
            counts_one=result.counts_one,
            shots=result.shots,
            method="swap_test",
            simulator=result.simulator,
            seed=call_seed,
        )
        self.num_estimates += 1
        if self.cache:
            self._cache[key] = estimate
        return estimate

    def _next_seed(self) -> int | None:
        if self.seed is None:
            return None
        return int(self._rng.integers(0, np.iinfo(np.uint32).max))


def _cache_key(
    vector_x: np.ndarray,
    vector_y: np.ndarray,
    *,
    shots: int,
    simulator: str,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(shots).encode("utf-8"))
    hasher.update(simulator.lower().encode("utf-8"))
    for vector in (vector_x, vector_y):
        values = np.ascontiguousarray(np.asarray(vector, dtype=np.float64).reshape(-1))
        hasher.update(str(values.shape).encode("utf-8"))
        hasher.update(values.tobytes())
    return hasher.hexdigest()
