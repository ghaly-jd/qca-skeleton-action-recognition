"""Common interface for quantum overlap estimators.

Step 1.16 profiling note: the global Q-SDTW path now uses a vectorized exact
overlap + binomial-sampling path for basis stacks. On MSR Action3D, the 60x40
rank=2 shots=1024 timing changed from 1.09s to 0.086s, and the full-data
single-seed rank=2 shots=1024 timing is 0.974s.
"""

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

    @property
    def supports_vectorized_sampling(self) -> bool:
        """Return whether this estimator can batch ideal SWAP-test sampling."""
        return self.simulator.lower() in {"exact", "sampling", "shot_sampling"}

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

    def estimate_basis_pair_squared_overlaps(
        self,
        basis_x: np.ndarray,
        basis_y: np.ndarray,
    ) -> np.ndarray:
        """Estimate all squared overlaps between two ``D x r`` bases."""
        x = _as_basis(basis_x, name="basis_x")
        y = _as_basis(basis_y, name="basis_y")
        if x.shape != y.shape:
            raise ValueError("Basis matrices must have matching D x r dimensions.")
        if not self.supports_vectorized_sampling:
            return self._estimate_basis_pair_scalar(x, y)

        overlaps = self.estimate_basis_stack_squared_overlaps(
            x[np.newaxis, ...],
            y[np.newaxis, ...],
        )
        return overlaps[0, 0]

    def estimate_basis_stack_squared_overlaps(
        self,
        test_bases: np.ndarray,
        train_bases: np.ndarray,
    ) -> np.ndarray:
        """Vectorize SWAP-test squared-overlap estimates for basis stacks.

        Parameters
        ----------
        test_bases, train_bases:
            Arrays with shape ``N x D x r`` and ``M x D x r``.

        Returns
        -------
        np.ndarray
            Estimated squared overlaps with shape ``N x M x r x r``.
        """
        if not self.supports_vectorized_sampling:
            raise ValueError(
                "Vectorized SWAP-test sampling supports only simulator='exact' "
                "or simulator='sampling'."
            )

        test = _as_basis_stack(test_bases, name="test_bases")
        train = _as_basis_stack(train_bases, name="train_bases")
        if test.shape[1:] != train.shape[1:]:
            raise ValueError("Basis stacks must have matching D x r dimensions.")

        exact_overlaps = _pairwise_squared_overlaps(test, train)
        simulator = self.simulator.lower()
        if simulator == "exact":
            estimates = exact_overlaps
        else:
            probability_zero = 0.5 * (1.0 + exact_overlaps)
            counts_zero = self._rng.binomial(self.shots, probability_zero)
            probability_zero_hat = counts_zero / self.shots
            estimates = np.clip(2.0 * probability_zero_hat - 1.0, 0.0, 1.0)

        self.num_estimates += int(estimates.size)
        return np.asarray(estimates, dtype=np.float64)

    def _next_seed(self) -> int | None:
        if self.seed is None:
            return None
        return int(self._rng.integers(0, np.iinfo(np.uint32).max))

    def _estimate_basis_pair_scalar(
        self,
        basis_x: np.ndarray,
        basis_y: np.ndarray,
    ) -> np.ndarray:
        rank = basis_x.shape[1]
        overlaps = np.empty((rank, rank), dtype=np.float64)
        for i in range(rank):
            for j in range(rank):
                overlaps[i, j] = self.estimate(basis_x[:, i], basis_y[:, j]).overlap_squared
        return overlaps


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


def _as_basis(values: np.ndarray, *, name: str) -> np.ndarray:
    basis = np.asarray(values, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"{name} must have shape D x r.")
    if basis.shape[1] <= 0:
        raise ValueError("Subspace rank must be positive.")
    return basis


def _as_basis_stack(values: np.ndarray, *, name: str) -> np.ndarray:
    stack = np.asarray(values, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"{name} must have shape N x D x r.")
    if stack.shape[2] <= 0:
        raise ValueError("Subspace rank must be positive.")
    return stack


def _pairwise_squared_overlaps(test_bases: np.ndarray, train_bases: np.ndarray) -> np.ndarray:
    test = _normalize_basis_stack(test_bases)
    train = _normalize_basis_stack(train_bases)
    overlaps = np.einsum("tdr,nds->tnrs", test, train, optimize=True)
    return np.clip(np.square(overlaps), 0.0, 1.0)


def _normalize_basis_stack(bases: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(bases, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Basis vectors must be non-zero.")
    return bases / norms
