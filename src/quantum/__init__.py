"""Quantum state preparation and overlap-estimation utilities."""

from src.quantum.overlap_estimation import OverlapEstimate, SwapTestOverlapEstimator
from src.quantum.state_preparation import AmplitudeEncodedState, amplitude_encode
from src.quantum.swap_test import SwapTestResult, estimate_squared_overlap_swap_test

__all__ = [
    "AmplitudeEncodedState",
    "OverlapEstimate",
    "SwapTestOverlapEstimator",
    "SwapTestResult",
    "amplitude_encode",
    "estimate_squared_overlap_swap_test",
]
