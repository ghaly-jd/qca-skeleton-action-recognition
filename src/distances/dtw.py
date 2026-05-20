"""Dynamic time warping distances for skeleton sequences."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Union

import numpy as np

try:  # pragma: no cover - exercised when numba is available in the runtime.
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path is covered by Python tests.
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False


WindowRatio = Union[float, int, str, None]


def dtw_distance(
    sequence_x: np.ndarray,
    sequence_y: np.ndarray,
    *,
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> float:
    """Return exact DTW distance between two ``T x D`` sequences.

    The optional ``window_ratio`` applies a Sakoe-Chiba band. Ratios are converted
    to a window in frames using ``ceil(max(T_x, T_y) * ratio)`` and widened when
    necessary so sequences of different lengths still have a feasible path.
    """
    x, y = _validate_pair(sequence_x, sequence_y)
    ratio = normalize_window_ratio(window_ratio)
    if _NUMBA_AVAILABLE:
        return float(_dtw_distance_numba(x, y, bool(normalize_by_path_length), ratio))
    return float(_dtw_distance_python(x, y, bool(normalize_by_path_length), ratio))


def pairwise_dtw_distances(
    test_sequences: Sequence[np.ndarray],
    train_sequences: Sequence[np.ndarray],
    *,
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
    backend: str = "auto",
    torch_device: str | None = None,
    torch_dtype: str = "float32",
) -> np.ndarray:
    """Return an ``N_test x N_train`` matrix of exact DTW distances."""
    if len(test_sequences) == 0:
        raise ValueError("test_sequences must not be empty.")
    if len(train_sequences) == 0:
        raise ValueError("train_sequences must not be empty.")

    test_padded, test_lengths = _pad_sequences(test_sequences)
    train_padded, train_lengths = _pad_sequences(train_sequences)
    if test_padded.shape[2] != train_padded.shape[2]:
        raise ValueError("Train and test sequences must have the same feature dimension.")

    ratio = normalize_window_ratio(window_ratio)
    backend = backend.lower()
    if backend == "auto":
        backend = "numba" if _NUMBA_AVAILABLE else "numpy"

    if backend == "numba":
        if not _NUMBA_AVAILABLE:
            raise RuntimeError("DTW backend 'numba' was requested, but numba is unavailable.")
        return np.asarray(
            _pairwise_dtw_numba(
                test_padded,
                test_lengths,
                train_padded,
                train_lengths,
                bool(normalize_by_path_length),
                ratio,
            ),
            dtype=np.float64,
        )

    if backend == "numpy":
        return _pairwise_dtw_vectorized(
            test_padded,
            test_lengths,
            train_padded,
            train_lengths,
            bool(normalize_by_path_length),
            ratio,
        )

    if backend in {"torch", "torch_cuda", "torch_cpu"}:
        return _pairwise_dtw_torch(
            test_padded,
            test_lengths,
            train_padded,
            train_lengths,
            bool(normalize_by_path_length),
            ratio,
            backend=backend,
            device=torch_device,
            dtype_name=torch_dtype,
        )

    raise ValueError(
        "Unknown DTW backend "
        f"'{backend}'. Known values: auto, numpy, numba, torch, torch_cpu, torch_cuda."
    )


def normalize_window_ratio(window_ratio: WindowRatio) -> float:
    """Normalize CLI/YAML window-ratio values to a numeric sentinel."""
    if window_ratio is None:
        return -1.0
    if isinstance(window_ratio, str):
        value = window_ratio.strip().lower()
        if value in {"", "none", "null", "false"}:
            return -1.0
        window_ratio = float(value)

    ratio = float(window_ratio)
    if not np.isfinite(ratio):
        raise ValueError("window_ratio must be finite.")
    if ratio == -1.0:
        return ratio
    if ratio < 0.0:
        raise ValueError("window_ratio must be non-negative or None.")
    return ratio


def window_ratio_label(window_ratio: WindowRatio) -> str:
    """Return a stable CSV label for a window-ratio value."""
    ratio = normalize_window_ratio(window_ratio)
    if ratio < 0.0:
        return "none"
    return f"{ratio:g}"


def _validate_pair(sequence_x: np.ndarray, sequence_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sequence_x, dtype=np.float64)
    y = np.asarray(sequence_y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("DTW inputs must be 2D T x D matrices.")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("DTW inputs must have at least one frame.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("DTW inputs must have the same feature dimension.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("DTW inputs must contain only finite values.")
    return x, y


def _pad_sequences(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
    if not arrays:
        raise ValueError("sequences must not be empty.")

    feature_dim = arrays[0].shape[1] if arrays[0].ndim == 2 else None
    lengths: list[int] = []
    for sequence in arrays:
        if sequence.ndim != 2:
            raise ValueError("Every sequence must be a 2D T x D matrix.")
        if sequence.shape[0] == 0:
            raise ValueError("Every sequence must have at least one frame.")
        if feature_dim is None or sequence.shape[1] != feature_dim:
            raise ValueError("Every sequence must have the same feature dimension.")
        if not np.isfinite(sequence).all():
            raise ValueError("Sequences must contain only finite values.")
        lengths.append(int(sequence.shape[0]))

    padded = np.zeros((len(arrays), max(lengths), int(feature_dim)), dtype=np.float64)
    for index, sequence in enumerate(arrays):
        padded[index, : sequence.shape[0], :] = sequence
    return padded, np.asarray(lengths, dtype=np.int64)


def _dtw_distance_python(
    x: np.ndarray,
    y: np.ndarray,
    normalize_by_path_length: bool,
    window_ratio: float,
) -> float:
    n_frames_x, feature_dim = x.shape
    n_frames_y = y.shape[0]
    window = _window_size(n_frames_x, n_frames_y, window_ratio)

    previous = np.full(n_frames_y + 1, np.inf, dtype=np.float64)
    current = np.full(n_frames_y + 1, np.inf, dtype=np.float64)
    previous_lengths = np.zeros(n_frames_y + 1, dtype=np.int64)
    current_lengths = np.zeros(n_frames_y + 1, dtype=np.int64)
    previous[0] = 0.0

    for i in range(1, n_frames_x + 1):
        current.fill(np.inf)
        current_lengths.fill(0)
        j_start = max(1, i - window)
        j_end = min(n_frames_y, i + window)
        for j in range(j_start, j_end + 1):
            frame_cost = 0.0
            for dim in range(feature_dim):
                difference = x[i - 1, dim] - y[j - 1, dim]
                frame_cost += difference * difference
            frame_cost = math.sqrt(frame_cost)

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

    distance = previous[n_frames_y]
    if normalize_by_path_length and previous_lengths[n_frames_y] > 0:
        distance /= previous_lengths[n_frames_y]
    return float(distance)


def _pairwise_dtw_vectorized(
    test_padded: np.ndarray,
    test_lengths: np.ndarray,
    train_padded: np.ndarray,
    train_lengths: np.ndarray,
    normalize_by_path_length: bool,
    window_ratio: float,
) -> np.ndarray:
    distances = np.empty((test_padded.shape[0], train_padded.shape[0]), dtype=np.float64)
    for test_index, test_length in enumerate(test_lengths):
        distances[test_index] = _dtw_one_to_many_vectorized(
            test_padded[test_index, :test_length, :],
            train_padded,
            train_lengths,
            normalize_by_path_length,
            window_ratio,
        )
    return distances


def _dtw_one_to_many_vectorized(
    test_sequence: np.ndarray,
    train_padded: np.ndarray,
    train_lengths: np.ndarray,
    normalize_by_path_length: bool,
    window_ratio: float,
) -> np.ndarray:
    n_train, max_train_length, _ = train_padded.shape
    test_length = test_sequence.shape[0]
    train_rows = np.arange(n_train)
    windows = _window_sizes_for_lengths(test_length, train_lengths, window_ratio)

    previous = np.full((n_train, max_train_length + 1), np.inf, dtype=np.float64)
    current = np.full((n_train, max_train_length + 1), np.inf, dtype=np.float64)
    previous_lengths = np.zeros((n_train, max_train_length + 1), dtype=np.int64)
    current_lengths = np.zeros((n_train, max_train_length + 1), dtype=np.int64)
    previous[:, 0] = 0.0

    for i in range(1, test_length + 1):
        current.fill(np.inf)
        current_lengths.fill(0)
        frame = test_sequence[i - 1]

        for j in range(1, max_train_length + 1):
            active = (j <= train_lengths) & (j >= i - windows) & (j <= i + windows)
            if not np.any(active):
                continue

            differences = train_padded[:, j - 1, :] - frame
            frame_costs = np.sqrt(np.sum(differences * differences, axis=1))

            best_costs = previous[:, j - 1].copy()
            best_lengths = previous_lengths[:, j - 1].copy()

            up_costs = previous[:, j]
            up_lengths = previous_lengths[:, j]
            use_up = (up_costs < best_costs) | (
                (up_costs == best_costs) & (up_lengths < best_lengths)
            )
            best_costs = np.where(use_up, up_costs, best_costs)
            best_lengths = np.where(use_up, up_lengths, best_lengths)

            left_costs = current[:, j - 1]
            left_lengths = current_lengths[:, j - 1]
            use_left = (left_costs < best_costs) | (
                (left_costs == best_costs) & (left_lengths < best_lengths)
            )
            best_costs = np.where(use_left, left_costs, best_costs)
            best_lengths = np.where(use_left, left_lengths, best_lengths)

            current[:, j] = np.where(active, frame_costs + best_costs, np.inf)
            current_lengths[:, j] = np.where(active, best_lengths + 1, 0)

        previous, current = current, previous
        previous_lengths, current_lengths = current_lengths, previous_lengths

    final_distances = previous[train_rows, train_lengths].copy()
    final_lengths = previous_lengths[train_rows, train_lengths]
    if normalize_by_path_length:
        valid = final_lengths > 0
        final_distances[valid] = final_distances[valid] / final_lengths[valid]
    return final_distances


def _window_sizes_for_lengths(
    test_length: int,
    train_lengths: np.ndarray,
    window_ratio: float,
) -> np.ndarray:
    if window_ratio < 0.0:
        return np.maximum(test_length, train_lengths)
    ratio_windows = np.ceil(np.maximum(test_length, train_lengths) * window_ratio).astype(
        np.int64
    )
    return np.maximum(np.abs(test_length - train_lengths), ratio_windows)


def _pairwise_dtw_torch(
    test_padded: np.ndarray,
    test_lengths: np.ndarray,
    train_padded: np.ndarray,
    train_lengths: np.ndarray,
    normalize_by_path_length: bool,
    window_ratio: float,
    *,
    backend: str,
    device: str | None,
    dtype_name: str,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for DTW torch backends.") from exc

    if backend == "torch_cuda":
        if device is None:
            device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DTW backend 'torch_cuda' was requested, but PyTorch cannot see CUDA."
            )
    elif backend == "torch_cpu":
        device = "cpu"
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = _torch_dtype(torch, dtype_name)
    torch_device = torch.device(device)
    _validate_torch_device_support(torch, torch_device)
    train_tensor = torch.as_tensor(train_padded, dtype=dtype, device=torch_device)
    train_lengths_tensor = torch.as_tensor(
        train_lengths,
        dtype=torch.long,
        device=torch_device,
    )

    rows: list[np.ndarray] = []
    for test_index, test_length in enumerate(test_lengths):
        test_tensor = torch.as_tensor(
            test_padded[test_index, : int(test_length), :],
            dtype=dtype,
            device=torch_device,
        )
        distances = _dtw_one_to_many_torch_antidiagonal(
            test_tensor,
            train_tensor,
            train_lengths_tensor,
            normalize_by_path_length,
            window_ratio,
        )
        rows.append(distances.detach().cpu().numpy().astype(np.float64, copy=False))

    return np.stack(rows, axis=0)


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


def _dtw_one_to_many_torch_antidiagonal(
    test_sequence: Any,
    train_padded: Any,
    train_lengths: Any,
    normalize_by_path_length: bool,
    window_ratio: float,
) -> Any:
    import torch

    n_train = train_padded.shape[0]
    test_length = int(test_sequence.shape[0])
    max_train_length = int(train_padded.shape[1])
    device = train_padded.device
    dtype = train_padded.dtype

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

        train_frames = train_padded[:, j_indices - 1, :]
        test_frames = test_sequence[i_indices - 1, :].unsqueeze(0)
        frame_costs = torch.sqrt(torch.sum((train_frames - test_frames) ** 2, dim=2))

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

        values = torch.where(active, frame_costs + best_costs, torch.full_like(frame_costs, float("inf")))
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


def _is_better_predecessor(
    candidate_cost: float,
    candidate_length: int,
    current_cost: float,
    current_length: int,
) -> bool:
    if candidate_cost < current_cost:
        return True
    return bool(candidate_cost == current_cost and candidate_length < current_length)


def _window_size(n_frames_x: int, n_frames_y: int, window_ratio: float) -> int:
    if window_ratio < 0.0:
        return max(n_frames_x, n_frames_y)
    ratio_window = int(math.ceil(max(n_frames_x, n_frames_y) * window_ratio))
    return max(abs(n_frames_x - n_frames_y), ratio_window)


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _window_size_numba(n_frames_x: int, n_frames_y: int, window_ratio: float) -> int:
        if window_ratio < 0.0:
            return max(n_frames_x, n_frames_y)
        ratio_window = int(math.ceil(max(n_frames_x, n_frames_y) * window_ratio))
        length_gap = abs(n_frames_x - n_frames_y)
        if ratio_window < length_gap:
            return length_gap
        return ratio_window

    @njit(cache=True)
    def _dtw_distance_numba(
        x: np.ndarray,
        y: np.ndarray,
        normalize_by_path_length: bool,
        window_ratio: float,
    ) -> float:
        n_frames_x = x.shape[0]
        n_frames_y = y.shape[0]
        feature_dim = x.shape[1]
        window = _window_size_numba(n_frames_x, n_frames_y, window_ratio)

        previous = np.empty(n_frames_y + 1, dtype=np.float64)
        current = np.empty(n_frames_y + 1, dtype=np.float64)
        previous_lengths = np.zeros(n_frames_y + 1, dtype=np.int64)
        current_lengths = np.zeros(n_frames_y + 1, dtype=np.int64)

        for j in range(n_frames_y + 1):
            previous[j] = np.inf
            current[j] = np.inf
        previous[0] = 0.0

        for i in range(1, n_frames_x + 1):
            for j in range(n_frames_y + 1):
                current[j] = np.inf
                current_lengths[j] = 0

            j_start = max(1, i - window)
            j_end = min(n_frames_y, i + window)
            for j in range(j_start, j_end + 1):
                frame_cost = 0.0
                for dim in range(feature_dim):
                    difference = x[i - 1, dim] - y[j - 1, dim]
                    frame_cost += difference * difference
                frame_cost = math.sqrt(frame_cost)

                best_cost = previous[j - 1]
                best_length = previous_lengths[j - 1]
                if previous[j] < best_cost or (
                    previous[j] == best_cost and previous_lengths[j] < best_length
                ):
                    best_cost = previous[j]
                    best_length = previous_lengths[j]
                if current[j - 1] < best_cost or (
                    current[j - 1] == best_cost and current_lengths[j - 1] < best_length
                ):
                    best_cost = current[j - 1]
                    best_length = current_lengths[j - 1]

                current[j] = frame_cost + best_cost
                current_lengths[j] = best_length + 1

            swap_costs = previous
            previous = current
            current = swap_costs
            swap_lengths = previous_lengths
            previous_lengths = current_lengths
            current_lengths = swap_lengths

        distance = previous[n_frames_y]
        if normalize_by_path_length and previous_lengths[n_frames_y] > 0:
            distance /= previous_lengths[n_frames_y]
        return distance

    @njit(parallel=True, cache=True)
    def _pairwise_dtw_numba(
        test_padded: np.ndarray,
        test_lengths: np.ndarray,
        train_padded: np.ndarray,
        train_lengths: np.ndarray,
        normalize_by_path_length: bool,
        window_ratio: float,
    ) -> np.ndarray:
        n_test = test_padded.shape[0]
        n_train = train_padded.shape[0]
        distances = np.empty((n_test, n_train), dtype=np.float64)

        for test_index in prange(n_test):
            test_length = test_lengths[test_index]
            test_sequence = test_padded[test_index, :test_length, :]
            for train_index in range(n_train):
                train_length = train_lengths[train_index]
                train_sequence = train_padded[train_index, :train_length, :]
                distances[test_index, train_index] = _dtw_distance_numba(
                    test_sequence,
                    train_sequence,
                    normalize_by_path_length,
                    window_ratio,
                )

        return distances

else:

    def _dtw_distance_numba(*args: Any, **kwargs: Any) -> float:
        raise RuntimeError("Numba is not available.")

    def _pairwise_dtw_numba(*args: Any, **kwargs: Any) -> np.ndarray:
        raise RuntimeError("Numba is not available.")
