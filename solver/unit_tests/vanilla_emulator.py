import numpy as np
from itertools import product
import json
import os


def infer_gate_rank(gate: np.ndarray, d: int) -> int:
    """
    Infers the number of qudits a gate acts on, based on its shape and local dimension `d`.
    Raises an error if the gate shape is not consistent with a square tensor product operator.
    """
    if gate.ndim % 2 != 0:
        raise ValueError(f"Gate must have even number of dimensions, got {gate.ndim}")
    rank = gate.ndim // 2
    expected_shape = (d,) * (2 * rank)
    if gate.shape != expected_shape:
        raise ValueError(
            f"Gate shape {gate.shape} does not match expected shape {expected_shape} for d={d}"
        )
    return rank


def apply_gate_einsum(
    state: np.ndarray, gate: np.ndarray, qudits: list[int], d: int
) -> np.ndarray:
    """
    Applies a gate to the given `state` tensor using einsum.
    - `state`: state tensor of shape (d, d, ..., d)
    - `gate`: unitary tensor of shape (d,...,d, d,...,d) with 2r dimensions
    - `qudits`: list of target qudit indices (length must match inferred rank)
    - `d`: local dimension (e.g., 2 for qubits, 3 for qutrits)
    """
    n = state.ndim
    r = infer_gate_rank(gate, d)
    if len(qudits) != r:
        raise ValueError(
            f"Expected {r} qudit indices for this gate, got {len(qudits)}: {qudits}"
        )
    if max(qudits) >= n or min(qudits) < 0:
        raise IndexError(f"Qudit indices {qudits} out of bounds for {n}-qudit system")

    # Create einsum indices
    state_in = [chr(ord("a") + i) for i in range(n)]
    state_out = state_in.copy()
    gate_in = [chr(ord("A") + i) for i in range(r)]
    gate_out = [chr(ord("A") + i + r) for i in range(r)]

    for i, q in enumerate(qudits):
        state_in[q] = gate_in[i]
        state_out[q] = gate_out[i]

    einsum_str = (
        f"{''.join(gate_out + gate_in)},"
        f"{''.join(state_in)}->"
        f"{''.join(state_out)}"
    )
    print("Einsum string:", einsum_str)
    return np.einsum(einsum_str, gate, state)


def parse_gate_einsum(
    instr: str, state: np.ndarray, gate_dict: dict, d: int
) -> np.ndarray:
    """
    Parses a gate instruction string and applies the corresponding gate.
    - `instr`: gate string like 'H_0' or 'CX_0_2'
    - `state`: current state tensor
    - `gate_dict`: dictionary of gate names to unitary arrays
    - `d`: local dimension
    """
    parts = instr.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid gate instruction: {instr}")
    name = parts[0]
    qudits = list(map(int, parts[1:]))
    if name not in gate_dict:
        raise KeyError(f"Gate '{name}' not found in gate_dict")
    gate = gate_dict[name]
    return apply_gate_einsum(state, gate, qudits, d)


def simulate_circuit_einsum(
    circuit: list[str], gate_dict: dict, d: int, n: int
) -> dict:
    """
    Simulates a single quantum circuit using einsum-based statevector evolution.
    Returns a dictionary mapping computational basis strings to probabilities.
    """
    shape = (d,) * n
    state = np.zeros(shape, dtype=complex)
    state[(0,) * n] = 1.0  # Start in |00...0>

    for instr in circuit:
        state = parse_gate_einsum(instr, state, gate_dict, d)

    flat_state = state.reshape(-1)
    probs = np.abs(flat_state) ** 2
    labels = ["".join(map(str, i)) for i in product(range(d), repeat=n)]
    return dict(zip(labels, probs))


def extract_qudit_indices(instr: str) -> list[int]:
    """
    Extracts target qudit indices from a gate string like 'CX_0_2'.
    """
    return list(map(int, instr.split("_")[1:]))


def infer_circuit_width(circuit: list[str]) -> int:
    """
    Infers the number of qudits required to simulate the given circuit.
    """
    max_idx = -1
    for instr in circuit:
        indices = extract_qudit_indices(instr)
        if indices:
            max_idx = max(max_idx, max(indices))
    return max_idx + 1  # indices are 0-based


def check_unitary(U: np.ndarray) -> bool:
    """
    Checks if a matrix is unitary.
    """
    if U.ndim == 2:  # single-qudit case
        V = np.copy(U)
    elif U.ndim == 4:  # two-qudit case
        V = np.reshape(U, (U.shape[0] * U.shape[1], U.shape[0] * U.shape[1]))
    else:
        raise ValueError(f"Invalid gate shape: {U.shape}")
    return np.allclose(V @ V.conj().T, np.eye(V.shape[0]))


def check_gate_dict(gate_dict: dict) -> None:
    """
    Checks if a gate dictionary is valid.
    """
    for gate in gate_dict:
        if not check_unitary(gate_dict[gate]):
            raise ValueError(f"Gate {gate} is not unitary")


def simulate_circs(circuits: list[list[str]], gate_dict: dict, d: int) -> list[dict]:
    """
    Simulates a list of quantum circuits and returns a list of dictionaries
    mapping basis states to probabilities.
    Each circuit may use a different number of qudits.
    """
    check_gate_dict(gate_dict)

    results = []
    for circuit in circuits:
        n = infer_circuit_width(circuit)
        probs = simulate_circuit_einsum(circuit, gate_dict, d, n)
        results.append(probs)
    return results
