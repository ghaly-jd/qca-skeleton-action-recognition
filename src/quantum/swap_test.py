"""SWAP-test circuits and shot-based squared-overlap estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.quantum.state_preparation import amplitude_encode_pair


@dataclass(frozen=True)
class SwapTestResult:
    """One SWAP-test squared-overlap estimate."""

    overlap_squared: float
    probability_zero: float
    counts_zero: int
    counts_one: int
    shots: int
    simulator: str
    seed: int | None = None

    @property
    def counts(self) -> dict[str, int]:
        """Return measurement counts keyed by measured ancilla bit."""
        return {"0": self.counts_zero, "1": self.counts_one}


def swap_test_zero_probability(vector_x: np.ndarray, vector_y: np.ndarray) -> float:
    """Return the ideal SWAP-test probability of measuring ancilla ``0``."""
    state_x, state_y = amplitude_encode_pair(vector_x, vector_y)
    overlap_squared = float(abs(np.vdot(state_x.amplitudes, state_y.amplitudes)) ** 2)
    probability_zero = 0.5 * (1.0 + overlap_squared)
    return float(np.clip(probability_zero, 0.0, 1.0))


def estimate_squared_overlap_swap_test(
    vector_x: np.ndarray,
    vector_y: np.ndarray,
    *,
    shots: int = 1024,
    simulator: str = "sampling",
    seed: int | None = None,
) -> SwapTestResult:
    """Estimate ``|<x|y>|^2`` from SWAP-test ancilla measurements."""
    if shots <= 0:
        raise ValueError("shots must be positive.")

    simulator = simulator.lower()
    if simulator == "exact":
        probability_zero = swap_test_zero_probability(vector_x, vector_y)
        overlap_squared = max(0.0, min(1.0, 2.0 * probability_zero - 1.0))
        counts_zero = int(round(probability_zero * shots))
        return SwapTestResult(
            overlap_squared=overlap_squared,
            probability_zero=probability_zero,
            counts_zero=counts_zero,
            counts_one=shots - counts_zero,
            shots=shots,
            simulator=simulator,
            seed=seed,
        )

    if simulator in {"sampling", "shot_sampling"}:
        probability_zero = swap_test_zero_probability(vector_x, vector_y)
        rng = np.random.default_rng(seed)
        counts_zero = int(rng.binomial(shots, probability_zero))
        probability_zero_hat = counts_zero / shots
        overlap_squared = float(np.clip(2.0 * probability_zero_hat - 1.0, 0.0, 1.0))
        return SwapTestResult(
            overlap_squared=overlap_squared,
            probability_zero=probability_zero_hat,
            counts_zero=counts_zero,
            counts_one=shots - counts_zero,
            shots=shots,
            simulator="sampling",
            seed=seed,
        )

    if simulator in {"aer", "qiskit_aer"}:
        counts = _run_aer_swap_test(vector_x, vector_y, shots=shots, seed=seed)
        counts_zero = int(counts.get("0", 0))
        counts_one = int(counts.get("1", 0))
        probability_zero_hat = counts_zero / shots
        overlap_squared = float(np.clip(2.0 * probability_zero_hat - 1.0, 0.0, 1.0))
        return SwapTestResult(
            overlap_squared=overlap_squared,
            probability_zero=probability_zero_hat,
            counts_zero=counts_zero,
            counts_one=counts_one,
            shots=shots,
            simulator="qiskit_aer",
            seed=seed,
        )

    raise ValueError("simulator must be one of: exact, sampling, qiskit_aer.")


def swap_test_circuit(vector_x: np.ndarray, vector_y: np.ndarray, *, measure: bool = True):
    """Build a Qiskit SWAP-test circuit for two amplitude-encoded vectors."""
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise RuntimeError("Qiskit is required to build SWAP-test circuits.") from exc

    state_x, state_y = amplitude_encode_pair(vector_x, vector_y)
    num_state_qubits = state_x.num_qubits
    num_qubits = 1 + (2 * num_state_qubits)
    circuit = QuantumCircuit(num_qubits, 1 if measure else 0)

    ancilla = 0
    x_qubits = list(range(1, 1 + num_state_qubits))
    y_qubits = list(range(1 + num_state_qubits, num_qubits))

    circuit.initialize(state_x.amplitudes, x_qubits)
    circuit.initialize(state_y.amplitudes, y_qubits)
    circuit.h(ancilla)
    for x_qubit, y_qubit in zip(x_qubits, y_qubits):
        circuit.cswap(ancilla, x_qubit, y_qubit)
    circuit.h(ancilla)
    if measure:
        circuit.measure(ancilla, 0)
    return circuit


def _run_aer_swap_test(
    vector_x: np.ndarray,
    vector_y: np.ndarray,
    *,
    shots: int,
    seed: int | None,
) -> dict[str, int]:
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise RuntimeError("qiskit-aer is required for simulator='qiskit_aer'.") from exc

    circuit = swap_test_circuit(vector_x, vector_y, measure=True)
    simulator = AerSimulator(seed_simulator=seed)
    result = simulator.run(circuit, shots=shots).result()
    return {str(key): int(value) for key, value in result.get_counts().items()}
