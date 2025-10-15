import pytest
import numpy as np
from vanilla_emulator import simulate_circs
from scipy.stats import unitary_group


def build_bv_circuit(s: str) -> list[str]:
    """
    Constructs a Bernstein–Vazirani circuit for hidden string `s`, e.g. "101".
    Returns a list of gate instructions like ["H_0", "Z_1", "H_0"]
    """
    n = len(s)
    circuit = []

    # Step 1: Apply H to all qubits
    circuit += [f"H_{i}" for i in range(n)]

    # Step 2: Oracle: apply Z to qubit i if s[i] == '1'
    for i, bit in enumerate(s):
        if bit == "1":
            circuit.append(f"Z_{i}")  # Z = phase flip for that qubit

    # Step 3: Apply H again to all qubits
    circuit += [f"H_{i}" for i in range(n)]

    return circuit


def test_bv():
    def test_bv_case(s: str):
        Z = np.array([[1, 0], [0, -1]])
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        gate_dict = {
            "H": H,
            "Z": Z,
        }

        circuit = build_bv_circuit(s)
        result = simulate_circs([circuit], gate_dict, d=2)[0]
        # Expected outcome is the string s with probability ~1
        max_state = max(result.items(), key=lambda x: x[1])
        assert max_state[0] == s and np.isclose(
            max_state[1], 1.0, atol=1e-10
        ), "BV test failed!"

    for s in ["0", "1", "01", "10", "101", "100", "001", "000", "111"]:
        # Run the test
        print(f"Running test for s={s}")
        test_bv_case(s)


def test_1qudit_circuit():
    """
    Tests a 1qutrit circuit with a random unitary gate U.
    |0> - U - M
    """
    for dim in [2, 3, 4]:
        U = unitary_group.rvs(dim)
        gate_dict = {
            "U": U,
        }
        circuits = [["U_0"]]
        result = simulate_circs(circuits, gate_dict, dim)

        # Construct the final state for comparison
        initial_state_1q = np.zeros(dim)
        initial_state_1q[0] = 1
        final_state_1q = U @ initial_state_1q
        check_probs = np.abs(final_state_1q) ** 2

        # Check the probabilities
        for ind in range(dim):
            label = str(ind)
            assert np.isclose(
                result[0][label], check_probs[ind], atol=1e-10
            ), f"1qutrit circuit test failed! {label}: {result[0][label]} {check_probs[ind]}"


def test_2qudit_circuit():
    """
    Test a 2qutrit circuit with random unitary gates: U1 and U2 are 1qutrit unitaries, V is a 2qutrit unitary.
    |0> - U1 - V - M
    |0> - U2 - V - M
    """
    for dim in [2, 3, 4]:
        U1 = unitary_group.rvs(dim)
        U2 = unitary_group.rvs(dim)
        V = unitary_group.rvs(dim**2).reshape((dim, dim, dim, dim))
        gate_dict = {
            "U1": U1,
            "U2": U2,
            "V": V,
        }
        circuits = [["U1_0", "U2_1", "V_0_1"]]
        result = simulate_circs(circuits, gate_dict, dim)

        # Construct the final state for comparison
        initial_state_1q = np.zeros(dim)
        initial_state_1q[0] = 1
        initial_state_2q = np.kron(initial_state_1q, initial_state_1q)
        final_state_2q = V.reshape(dim**2, dim**2) @ np.kron(U1, U2) @ initial_state_2q
        check_probs = np.abs(final_state_2q) ** 2

        # Check the probabilities
        for ind in range(dim**2):
            q0_label = str(ind // dim)
            q1_label = str(ind % dim)
            label = q0_label + q1_label
            assert np.isclose(
                result[0][label], check_probs[ind], atol=1e-10
            ), f"2qutrit circuit test failed! {label}: {result[0][label]} {check_probs[ind]}"


def test_1qudit_circuits_from_ivan():
    gate_dict = {
        "Rx": np.array(
            [
                [0.70710678 + 0.0j, -0.0 - 0.70710678j, 0.0 + 0.0j],
                [-0.0 - 0.70710678j, 0.70710678 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ]
        ),
        "M": np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ]
        ),
        "X": np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        "Ry": np.array(
            [
                [0.70710678 + 0.0j, 0.70710678 + 0.0j, 0.0 + 0.0j],
                [-0.70710678 + 0.0j, 0.70710678 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ]
        ),
    }
    circuits = [
        ["Ry_0", "Ry_0", "M_0"],
        ["Ry_0", "Ry_0", "X_0", "M_0"],
        ["Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["X_0", "Ry_0", "Ry_0", "M_0"],
        ["X_0", "Ry_0", "Ry_0", "X_0", "M_0"],
        ["X_0", "Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["Rx_0", "Ry_0", "Ry_0", "M_0"],
        ["Rx_0", "Ry_0", "Ry_0", "X_0", "M_0"],
        ["Rx_0", "Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["Rx_0", "X_0", "Ry_0", "Ry_0", "M_0"],
        ["Rx_0", "X_0", "Ry_0", "Ry_0", "X_0", "M_0"],
        ["Rx_0", "X_0", "Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["Rx_0", "X_0", "X_0", "Ry_0", "Ry_0", "M_0"],
        ["Rx_0", "X_0", "X_0", "Ry_0", "Ry_0", "X_0", "M_0"],
        ["Rx_0", "X_0", "X_0", "Ry_0", "Ry_0", "X_0", "M_0"],
        ["Rx_0", "X_0", "X_0", "Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["Ry_0", "Ry_0", "M_0"],
        ["Ry_0", "Ry_0", "Rx_0", "M_0"],
        ["Ry_0", "Ry_0", "Rx_0", "M_0"],
        ["Ry_0", "Ry_0", "Rx_0", "X_0", "M_0"],
        ["Ry_0", "Ry_0", "Rx_0", "X_0", "X_0", "M_0"],
        ["Ry_0", "Ry_0", "Rx_0", "X_0", "X_0", "M_0"],
        ["Ry_0", "Ry_0", "M_0"],
        ["Ry_0", "Rx_0", "Ry_0", "M_0"],
        ["Ry_0", "Ry_0", "Ry_0", "M_0"],
        ["Ry_0", "Ry_0", "X_0", "M_0"],
        ["Ry_0", "Rx_0", "Ry_0", "X_0", "M_0"],
        ["Ry_0", "Ry_0", "Ry_0", "X_0", "M_0"],
        ["Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["Ry_0", "Rx_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["Ry_0", "Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["Ry_0", "X_0", "Ry_0", "Ry_0", "M_0"],
        ["Ry_0", "X_0", "Ry_0", "Ry_0", "X_0", "M_0"],
        ["Ry_0", "X_0", "Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
        ["Ry_0", "X_0", "X_0", "X_0", "Ry_0", "M_0"],
        ["Ry_0", "X_0", "X_0", "Ry_0", "Ry_0", "M_0"],
        ["Ry_0", "X_0", "X_0", "X_0", "Ry_0", "X_0", "M_0"],
        ["Ry_0", "X_0", "X_0", "Ry_0", "Ry_0", "X_0", "M_0"],
        ["Ry_0", "X_0", "X_0", "Ry_0", "Ry_0", "X_0", "X_0", "M_0"],
    ]
    result = simulate_circs(circuits, gate_dict, 3)

    def very_naive_simulate_circs(circuits, gate_dict, dim):
        rslts = []
        for circuit in circuits:
            psi = np.zeros(dim)
            psi[0] = 1
            for gate in circuit:
                gate_name = gate.split("_")[0]
                if gate_name in gate_dict:
                    if gate_name in gate_dict:
                        U = gate_dict[gate_name]
                        psi = U @ psi
                    else:
                        raise ValueError(f"Gate {gate_name} not found in gate_dict")
            probs = np.abs(psi) ** 2
            rslt_dict = {}
            for b in range(dim):
                rslt_dict[str(b)] = probs[b]
            rslts.append(rslt_dict)
            assert probs_sum_to_one(
                rslt_dict
            ), f"Probs sum to one failed! {rslt_dict} {circuit=}"

        return rslts

    def probs_sum_to_one(probs: dict):
        s = 0
        for outcome in probs:
            s += probs[outcome]
        return np.isclose(s, 1.0, atol=1e-10)

    naive_result = very_naive_simulate_circs(circuits, gate_dict, dim=3)
    for i in range(len(circuits)):
        assert result[i] == naive_result[i], f"Circuit {circuits[i]} failed!"
        assert probs_sum_to_one(result[i]), f"Probs sum to one failed! {result[i]}"
