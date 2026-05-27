"""SWAP-estimated Local Subspace-DTW distances."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.distances.dtw import WindowRatio
from src.distances.local_sdtw import (
    dtw_distance_from_cost_matrix,
    local_subspace_cost_matrix,
)
from src.distances.quantum_estimated_angles import quantum_subspace_affinity_distance
from src.features.local_subspace import LocalSubspaceSequence
from src.quantum.overlap_estimation import SwapTestOverlapEstimator


@dataclass(frozen=True)
class QuantumLocalSdtwPairDiagnostic:
    """Exact-vs-SWAP diagnostics for one test/train Local-SDTW pair."""

    test_index: int
    train_index: int
    test_sequence_id: str
    train_sequence_id: str
    exact_distance: float
    quantum_distance: float
    distance_abs_error: float
    cost_matrix_mae: float
    cost_matrix_rmse: float
    cost_matrix_max_abs_error: float
    num_local_costs: int


@dataclass(frozen=True)
class QuantumLocalSdtwComparison:
    """Pairwise exact and SWAP-estimated Local-SDTW distances plus diagnostics."""

    exact_distances: np.ndarray
    quantum_distances: np.ndarray
    pair_diagnostics: list[QuantumLocalSdtwPairDiagnostic]

    @property
    def num_dtw_pairs(self) -> int:
        return int(self.exact_distances.size)

    @property
    def num_local_costs(self) -> int:
        return int(sum(item.num_local_costs for item in self.pair_diagnostics))


def quantum_local_subspace_cost(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    *,
    estimator: SwapTestOverlapEstimator,
    affinity_normalization: str = "projection_frobenius",
) -> float:
    """Return a SWAP-estimated projection-affinity local subspace cost."""
    return quantum_subspace_affinity_distance(
        basis_x,
        basis_y,
        estimator=estimator,
        normalization=affinity_normalization,
    )


def quantum_local_subspace_cost_matrix(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    estimator: SwapTestOverlapEstimator,
    affinity_normalization: str = "projection_frobenius",
) -> np.ndarray:
    """Return a SWAP-estimated local cost matrix for two local subspace sequences."""
    _validate_local_sequence(sequence_x, name="sequence_x")
    _validate_local_sequence(sequence_y, name="sequence_y")

    costs = np.empty((len(sequence_x.windows), len(sequence_y.windows)), dtype=np.float64)
    for i, window_x in enumerate(sequence_x.windows):
        for j, window_y in enumerate(sequence_y.windows):
            costs[i, j] = quantum_local_subspace_cost(
                window_x.basis,
                window_y.basis,
                estimator=estimator,
                affinity_normalization=affinity_normalization,
            )
    return np.clip(costs, 0.0, 1.0)


def quantum_local_sdtw_distance(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    estimator: SwapTestOverlapEstimator,
    affinity_normalization: str = "projection_frobenius",
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> float:
    """Return SWAP-estimated Local Subspace-DTW distance."""
    costs = quantum_local_subspace_cost_matrix(
        sequence_x,
        sequence_y,
        estimator=estimator,
        affinity_normalization=affinity_normalization,
    )
    return dtw_distance_from_cost_matrix(
        costs,
        normalize_by_path_length=normalize_by_path_length,
        window_ratio=window_ratio,
    )


def pairwise_quantum_local_sdtw_distances(
    test_sequences: Sequence[LocalSubspaceSequence],
    train_sequences: Sequence[LocalSubspaceSequence],
    *,
    estimator: SwapTestOverlapEstimator,
    affinity_normalization: str = "projection_frobenius",
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> np.ndarray:
    """Return an ``N_test x N_train`` matrix of SWAP-estimated Local-SDTW distances."""
    _validate_sequence_collection(test_sequences, name="test_sequences")
    _validate_sequence_collection(train_sequences, name="train_sequences")

    distances = np.empty((len(test_sequences), len(train_sequences)), dtype=np.float64)
    for test_index, test_sequence in enumerate(test_sequences):
        for train_index, train_sequence in enumerate(train_sequences):
            distances[test_index, train_index] = quantum_local_sdtw_distance(
                test_sequence,
                train_sequence,
                estimator=estimator,
                affinity_normalization=affinity_normalization,
                normalize_by_path_length=normalize_by_path_length,
                window_ratio=window_ratio,
            )
    return distances


def compare_quantum_exact_local_sdtw_pair(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    test_index: int,
    train_index: int,
    estimator: SwapTestOverlapEstimator,
    affinity_normalization: str = "projection_frobenius",
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> QuantumLocalSdtwPairDiagnostic:
    """Compare exact and SWAP-estimated Local-SDTW for one sequence pair."""
    exact_costs = local_subspace_cost_matrix(
        sequence_x,
        sequence_y,
        metric="projection_affinity",
        affinity_normalization=affinity_normalization,
    )
    quantum_costs = quantum_local_subspace_cost_matrix(
        sequence_x,
        sequence_y,
        estimator=estimator,
        affinity_normalization=affinity_normalization,
    )
    exact_distance = dtw_distance_from_cost_matrix(
        exact_costs,
        normalize_by_path_length=normalize_by_path_length,
        window_ratio=window_ratio,
    )
    quantum_distance = dtw_distance_from_cost_matrix(
        quantum_costs,
        normalize_by_path_length=normalize_by_path_length,
        window_ratio=window_ratio,
    )
    cost_error = quantum_costs - exact_costs
    return QuantumLocalSdtwPairDiagnostic(
        test_index=int(test_index),
        train_index=int(train_index),
        test_sequence_id=sequence_x.sequence_id,
        train_sequence_id=sequence_y.sequence_id,
        exact_distance=float(exact_distance),
        quantum_distance=float(quantum_distance),
        distance_abs_error=float(abs(quantum_distance - exact_distance)),
        cost_matrix_mae=float(np.mean(np.abs(cost_error))),
        cost_matrix_rmse=float(np.sqrt(np.mean(np.square(cost_error)))),
        cost_matrix_max_abs_error=float(np.max(np.abs(cost_error))),
        num_local_costs=int(exact_costs.size),
    )


def pairwise_quantum_exact_local_sdtw_comparison(
    test_sequences: Sequence[LocalSubspaceSequence],
    train_sequences: Sequence[LocalSubspaceSequence],
    *,
    estimator: SwapTestOverlapEstimator,
    affinity_normalization: str = "projection_frobenius",
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> QuantumLocalSdtwComparison:
    """Compare exact and SWAP-estimated Local-SDTW for every test/train pair."""
    _validate_sequence_collection(test_sequences, name="test_sequences")
    _validate_sequence_collection(train_sequences, name="train_sequences")

    exact_distances = np.empty((len(test_sequences), len(train_sequences)), dtype=np.float64)
    quantum_distances = np.empty_like(exact_distances)
    diagnostics: list[QuantumLocalSdtwPairDiagnostic] = []

    for test_index, test_sequence in enumerate(test_sequences):
        for train_index, train_sequence in enumerate(train_sequences):
            diagnostic = compare_quantum_exact_local_sdtw_pair(
                test_sequence,
                train_sequence,
                test_index=test_index,
                train_index=train_index,
                estimator=estimator,
                affinity_normalization=affinity_normalization,
                normalize_by_path_length=normalize_by_path_length,
                window_ratio=window_ratio,
            )
            exact_distances[test_index, train_index] = diagnostic.exact_distance
            quantum_distances[test_index, train_index] = diagnostic.quantum_distance
            diagnostics.append(diagnostic)

    return QuantumLocalSdtwComparison(
        exact_distances=exact_distances,
        quantum_distances=quantum_distances,
        pair_diagnostics=diagnostics,
    )


def _validate_sequence_collection(
    sequences: Sequence[LocalSubspaceSequence],
    *,
    name: str,
) -> None:
    if len(sequences) == 0:
        raise ValueError(f"{name} must not be empty.")
    for sequence in sequences:
        _validate_local_sequence(sequence, name=name)


def _validate_local_sequence(sequence: LocalSubspaceSequence, *, name: str) -> None:
    if not sequence.windows:
        raise ValueError(f"{name} must contain at least one local window.")
