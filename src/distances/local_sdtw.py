"""Dynamic time warping over sequences of local motion subspaces."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.distances.dtw import WindowRatio, normalize_window_ratio
from src.distances.quantum_estimated_angles import exact_subspace_affinity_distance
from src.distances.subspace_distances import DISTANCE_NAMES, subspace_distance
from src.features.local_subspace import LocalSubspaceSequence


LOCAL_SUBSPACE_DISTANCE_NAMES = ["projection_affinity", *DISTANCE_NAMES]
LOCAL_SDTW_BACKENDS = ["numpy", "torch", "torch_cpu", "torch_cuda"]


def local_subspace_cost(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    *,
    metric: str = "projection_affinity",
    affinity_normalization: str = "projection_frobenius",
) -> float:
    """Return the local cost between two ``D x r`` subspace bases."""
    metric = metric.lower()
    if metric == "projection_affinity":
        return exact_subspace_affinity_distance(
            basis_x,
            basis_y,
            normalization=affinity_normalization,
        )
    if metric in DISTANCE_NAMES:
        return subspace_distance(basis_x, basis_y, metric=metric)

    known = ", ".join(LOCAL_SUBSPACE_DISTANCE_NAMES)
    raise ValueError(f"Unknown local subspace distance '{metric}'. Known values: {known}.")


def local_subspace_cost_matrix(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    metric: str = "projection_affinity",
    affinity_normalization: str = "projection_frobenius",
) -> np.ndarray:
    """Return an ``M x N`` local subspace cost matrix."""
    if not sequence_x.windows:
        raise ValueError("sequence_x must contain at least one local window.")
    if not sequence_y.windows:
        raise ValueError("sequence_y must contain at least one local window.")

    costs = np.empty((len(sequence_x.windows), len(sequence_y.windows)), dtype=np.float64)
    for i, window_x in enumerate(sequence_x.windows):
        for j, window_y in enumerate(sequence_y.windows):
            costs[i, j] = local_subspace_cost(
                window_x.basis,
                window_y.basis,
                metric=metric,
                affinity_normalization=affinity_normalization,
            )
    return costs


def local_sdtw_distance(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    metric: str = "projection_affinity",
    affinity_normalization: str = "projection_frobenius",
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> float:
    """Return Local Subspace-DTW distance between two local subspace sequences."""
    costs = local_subspace_cost_matrix(
        sequence_x,
        sequence_y,
        metric=metric,
        affinity_normalization=affinity_normalization,
    )
    return dtw_distance_from_cost_matrix(
        costs,
        normalize_by_path_length=normalize_by_path_length,
        window_ratio=window_ratio,
    )


def pairwise_local_sdtw_distances(
    test_sequences: Sequence[LocalSubspaceSequence],
    train_sequences: Sequence[LocalSubspaceSequence],
    *,
    metric: str = "projection_affinity",
    affinity_normalization: str = "projection_frobenius",
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
    backend: str = "numpy",
    torch_device: str | None = None,
    torch_dtype: str = "float32",
) -> np.ndarray:
    """Return an ``N_test x N_train`` matrix of exact Local-SDTW distances."""
    if len(test_sequences) == 0:
        raise ValueError("test_sequences must not be empty.")
    if len(train_sequences) == 0:
        raise ValueError("train_sequences must not be empty.")

    backend = backend.lower()
    if backend in {"torch", "torch_cpu", "torch_cuda"}:
        return _pairwise_local_sdtw_torch(
            test_sequences,
            train_sequences,
            metric=metric,
            affinity_normalization=affinity_normalization,
            normalize_by_path_length=normalize_by_path_length,
            window_ratio=window_ratio,
            backend=backend,
            device=torch_device,
            dtype_name=torch_dtype,
        )
    if backend != "numpy":
        known = ", ".join(LOCAL_SDTW_BACKENDS)
        raise ValueError(f"Unknown Local-SDTW backend '{backend}'. Known values: {known}.")

    distances = np.empty((len(test_sequences), len(train_sequences)), dtype=np.float64)
    for test_index, test_sequence in enumerate(test_sequences):
        for train_index, train_sequence in enumerate(train_sequences):
            distances[test_index, train_index] = local_sdtw_distance(
                test_sequence,
                train_sequence,
                metric=metric,
                affinity_normalization=affinity_normalization,
                normalize_by_path_length=normalize_by_path_length,
                window_ratio=window_ratio,
            )
    return distances


def _pairwise_local_sdtw_torch(
    test_sequences: Sequence[LocalSubspaceSequence],
    train_sequences: Sequence[LocalSubspaceSequence],
    *,
    metric: str,
    affinity_normalization: str,
    normalize_by_path_length: bool,
    window_ratio: WindowRatio,
    backend: str,
    device: str | None,
    dtype_name: str,
) -> np.ndarray:
    if metric.lower() != "projection_affinity":
        raise ValueError(
            "Torch Local-SDTW currently supports only metric='projection_affinity'."
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Local-SDTW torch backends.") from exc

    if backend == "torch_cuda":
        if device is None:
            device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Local-SDTW backend 'torch_cuda' was requested, "
                "but PyTorch cannot see CUDA."
            )
    elif backend == "torch_cpu":
        device = "cpu"
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = _torch_dtype(torch, dtype_name)
    torch_device = torch.device(device)
    _validate_torch_device_support(torch, torch_device)

    train_padded, train_lengths = _pad_local_bases(train_sequences)
    train_tensor = torch.as_tensor(train_padded, dtype=dtype, device=torch_device)
    train_lengths_tensor = torch.as_tensor(
        train_lengths,
        dtype=torch.long,
        device=torch_device,
    )
    ratio = normalize_window_ratio(window_ratio)

    rows: list[np.ndarray] = []
    for test_sequence in test_sequences:
        test_bases = _stack_local_bases(test_sequence)
        test_tensor = torch.as_tensor(test_bases, dtype=dtype, device=torch_device)
        distances = _local_sdtw_one_to_many_torch(
            test_tensor,
            train_tensor,
            train_lengths_tensor,
            normalize_by_path_length=bool(normalize_by_path_length),
            window_ratio=ratio,
            affinity_normalization=affinity_normalization,
        )
        rows.append(distances.detach().cpu().numpy().astype(np.float64, copy=False))

    return np.stack(rows, axis=0)


def _local_sdtw_one_to_many_torch(
    test_bases: Any,
    train_padded: Any,
    train_lengths: Any,
    *,
    normalize_by_path_length: bool,
    window_ratio: float,
    affinity_normalization: str,
) -> Any:
    import torch

    n_train = int(train_padded.shape[0])
    test_length = int(test_bases.shape[0])
    max_train_length = int(train_padded.shape[1])
    device = train_padded.device
    dtype = train_padded.dtype

    local_costs = _projection_affinity_cost_tensor_torch(
        test_bases,
        train_padded,
        normalization=affinity_normalization,
    )

    costs = torch.full(
        (n_train, test_length + 1, max_train_length + 1),
        float("inf"),
        dtype=dtype,
        device=device,
    )
    path_lengths = torch.zeros(
        (n_train, test_length + 1, max_train_length + 1),
        dtype=torch.long,
        device=device,
    )
    costs[:, 0, 0] = 0.0

    train_rows = torch.arange(n_train, device=device)
    windows = _window_sizes_for_lengths_torch(test_length, train_lengths, window_ratio)

    for diagonal in range(2, test_length + max_train_length + 1):
        i_start = max(1, diagonal - max_train_length)
        i_end = min(test_length, diagonal - 1)
        if i_start > i_end:
            continue

        i_indices = torch.arange(i_start, i_end + 1, dtype=torch.long, device=device)
        j_indices = diagonal - i_indices
        active = (
            (j_indices.unsqueeze(0) <= train_lengths.unsqueeze(1))
            & (j_indices.unsqueeze(0) >= i_indices.unsqueeze(0) - windows.unsqueeze(1))
            & (j_indices.unsqueeze(0) <= i_indices.unsqueeze(0) + windows.unsqueeze(1))
        )

        frame_costs = local_costs[:, i_indices - 1, j_indices - 1]
        diagonal_costs = costs[:, i_indices - 1, j_indices - 1]
        diagonal_lengths = path_lengths[:, i_indices - 1, j_indices - 1]
        up_costs = costs[:, i_indices - 1, j_indices]
        up_lengths = path_lengths[:, i_indices - 1, j_indices]
        left_costs = costs[:, i_indices, j_indices - 1]
        left_lengths = path_lengths[:, i_indices, j_indices - 1]

        best_costs = diagonal_costs
        best_lengths = diagonal_lengths
        use_up = (up_costs < best_costs) | (
            (up_costs == best_costs) & (up_lengths < best_lengths)
        )
        best_costs = torch.where(use_up, up_costs, best_costs)
        best_lengths = torch.where(use_up, up_lengths, best_lengths)
        use_left = (left_costs < best_costs) | (
            (left_costs == best_costs) & (left_lengths < best_lengths)
        )
        best_costs = torch.where(use_left, left_costs, best_costs)
        best_lengths = torch.where(use_left, left_lengths, best_lengths)

        values = torch.where(
            active,
            frame_costs + best_costs,
            torch.full_like(frame_costs, float("inf")),
        )
        lengths = torch.where(active, best_lengths + 1, torch.zeros_like(best_lengths))
        costs[:, i_indices, j_indices] = values
        path_lengths[:, i_indices, j_indices] = lengths

    final_costs = costs[train_rows, test_length, train_lengths].clone()
    final_lengths = path_lengths[train_rows, test_length, train_lengths]
    if normalize_by_path_length:
        valid = final_lengths > 0
        final_costs = torch.where(
            valid,
            final_costs / final_lengths.to(dtype=final_costs.dtype),
            final_costs,
        )
    return final_costs


def _projection_affinity_cost_tensor_torch(
    test_bases: Any,
    train_padded: Any,
    *,
    normalization: str,
) -> Any:
    import torch

    rank = int(test_bases.shape[2])
    overlaps = torch.einsum("idr,njds->nijrs", test_bases, train_padded)
    squared_overlaps = overlaps.square()

    normalization = normalization.lower()
    if normalization == "projection_frobenius":
        affinity = squared_overlaps.sum(dim=(-2, -1)) / rank
    elif normalization == "mean":
        affinity = squared_overlaps.mean(dim=(-2, -1))
    else:
        raise ValueError(f"Unknown affinity normalization '{normalization}'.")
    affinity = torch.clamp(affinity, min=0.0, max=1.0)
    return torch.clamp(1.0 - affinity, min=0.0)


def _pad_local_bases(
    sequences: Sequence[LocalSubspaceSequence],
) -> tuple[np.ndarray, np.ndarray]:
    stacks = [_stack_local_bases(sequence) for sequence in sequences]
    if not stacks:
        raise ValueError("sequences must not be empty.")

    basis_shape = stacks[0].shape[1:]
    lengths = []
    for stack in stacks:
        if stack.shape[1:] != basis_shape:
            raise ValueError("All local subspace bases must have matching D x r shapes.")
        lengths.append(int(stack.shape[0]))

    padded = np.zeros((len(stacks), max(lengths), *basis_shape), dtype=np.float64)
    for index, stack in enumerate(stacks):
        padded[index, : stack.shape[0], :, :] = stack
    return padded, np.asarray(lengths, dtype=np.int64)


def _stack_local_bases(sequence: LocalSubspaceSequence) -> np.ndarray:
    if not sequence.windows:
        raise ValueError("Local subspace sequences must contain at least one window.")
    bases = [np.asarray(window.basis, dtype=np.float64) for window in sequence.windows]
    basis_shape = bases[0].shape
    for basis in bases:
        if basis.shape != basis_shape:
            raise ValueError("All windows in a local sequence must share D x r shape.")
        if basis.ndim != 2 or basis.shape[1] <= 0:
            raise ValueError("Local window bases must have shape D x r with r > 0.")
        if not np.isfinite(basis).all():
            raise ValueError("Local window bases must contain only finite values.")
    return np.stack(bases, axis=0)


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    dtype = dtype_name.lower()
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("torch_dtype must be 'float32' or 'float64'.")


def _validate_torch_device_support(torch: Any, device: Any) -> None:
    if device.type != "cuda":
        return

    capability = torch.cuda.get_device_capability(device)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and required_arch not in supported_arches:
        supported = " ".join(sorted(supported_arches))
        raise RuntimeError(
            "PyTorch can see this CUDA device, but the installed PyTorch build does "
            f"not include kernels for {required_arch}. Supported arches: {supported}. "
            "Install a newer CUDA-enabled PyTorch build, then rerun with "
            "--backend torch_cuda."
        )


def _window_sizes_for_lengths_torch(
    test_length: int,
    train_lengths: Any,
    window_ratio: float,
) -> Any:
    import torch

    if window_ratio < 0.0:
        return torch.maximum(
            torch.full_like(train_lengths, test_length),
            train_lengths,
        )
    ratio_windows = torch.ceil(
        torch.maximum(
            torch.full_like(train_lengths, test_length),
            train_lengths,
        ).to(dtype=torch.float64)
        * float(window_ratio)
    ).to(dtype=torch.long)
    return torch.maximum(torch.abs(train_lengths - test_length), ratio_windows)


def dtw_distance_from_cost_matrix(
    cost_matrix: np.ndarray,
    *,
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> float:
    """Run exact DTW over a precomputed local-cost matrix."""
    costs = _as_cost_matrix(cost_matrix)
    num_rows, num_cols = costs.shape
    ratio = normalize_window_ratio(window_ratio)
    window = _window_size(num_rows, num_cols, ratio)

    previous = np.full(num_cols + 1, np.inf, dtype=np.float64)
    current = np.full(num_cols + 1, np.inf, dtype=np.float64)
    previous_lengths = np.zeros(num_cols + 1, dtype=np.int64)
    current_lengths = np.zeros(num_cols + 1, dtype=np.int64)
    previous[0] = 0.0

    for i in range(1, num_rows + 1):
        current.fill(np.inf)
        current_lengths.fill(0)
        j_start = max(1, i - window)
        j_end = min(num_cols, i + window)
        for j in range(j_start, j_end + 1):
            frame_cost = costs[i - 1, j - 1]

            best_cost = previous[j - 1]
            best_length = previous_lengths[j - 1]
            if _is_better_predecessor(previous[j], previous_lengths[j], best_cost, best_length):
                best_cost = previous[j]
                best_length = previous_lengths[j]
            if _is_better_predecessor(current[j - 1], current_lengths[j - 1], best_cost, best_length):
                best_cost = current[j - 1]
                best_length = current_lengths[j - 1]

            current[j] = frame_cost + best_cost
            current_lengths[j] = best_length + 1

        previous, current = current, previous
        previous_lengths, current_lengths = current_lengths, previous_lengths

    distance = previous[num_cols]
    if normalize_by_path_length and previous_lengths[num_cols] > 0:
        distance /= previous_lengths[num_cols]
    return float(distance)


def _as_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be a 2D matrix.")
    if costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError("cost_matrix must be non-empty.")
    if not np.isfinite(costs).all():
        raise ValueError("cost_matrix must contain only finite values.")
    return costs


def _window_size(num_rows: int, num_cols: int, window_ratio: float) -> int:
    if window_ratio < 0.0:
        return max(num_rows, num_cols)
    requested = int(math.ceil(max(num_rows, num_cols) * window_ratio))
    return max(requested, abs(num_rows - num_cols))


def _is_better_predecessor(
    candidate_cost: float,
    candidate_length: Any,
    best_cost: float,
    best_length: Any,
) -> bool:
    if candidate_cost < best_cost:
        return True
    if candidate_cost > best_cost:
        return False
    return int(candidate_length) < int(best_length)
