import pytest
import numpy as np
from solver.unit_tests.vanilla_emulator import simulate_circs
from scipy.stats import unitary_group
import solver.utils.channel_utils as c_util
import solver.circuits_generation as cg
from solver.QCCalc import QCEvaluator
from solver.utils.misc import NconTemplate


def simple_template(tmpl: NconTemplate, n, single_qud_gates_names) -> NconTemplate:
    """
    Converts a template made for hidden or estimator evaluators into a template suited for pure evaluator.
    This conversion changes only tensors IDs, leaving tensor network structure intact.
    """
    tensors, net_struc, con_order, out_order = tmpl
    new_tensors = tensors.copy()
    e_id = len(single_qud_gates_names) * n

    for idx, old_tensor_id in enumerate(tensors):
        if old_tensor_id == e_id:
            new_tensors[idx] = len(single_qud_gates_names)
        elif old_tensor_id < e_id:
            new_tensors[idx] = old_tensor_id // n
        else:
            shifted = old_tensor_id - e_id - 1
            new_tensors[idx] = shifted // (n * (n - 1))
            new_tensors[idx] += len(single_qud_gates_names)

    new_template = [new_tensors, net_struc, con_order, out_order]
    return new_template


def compare_results(vanilla_res, tomography_output, err=0.05):
    for idx in range(len(vanilla_res)):
        res_keys_raw = np.unique(tomography_output, axis=0)
        tom_dict = {''.join(map(str, i)): 0 for i in res_keys_raw}
        for i in tomography_output.numpy():
            key = ''.join(map(str, i))
            tom_dict[key] += 1
        tom_total = len(tomography_output)
        tom_res = {i: tom_dict[i]/tom_total for i in tom_dict.keys()}
        for i in tom_res.keys():
            assert np.isclose(tom_res[i], vanilla_res[idx][i], atol=err), f'Error at circuit {idx} for value {i}'
        return tom_res


def test_simple_1q_circuit():
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    gate_dict = {
        "H": H
    }
    dim = 2
    circuits = [["H_0"]]
    vanilla_res = simulate_circs(circuits, gate_dict, dim)

    pure_channels = {
        "H": c_util.convert_1qmatrix_to_channel(H)
    }
    gen = cg.DataGenerator(qubits_num=1,
                           gates_names=["H"],
                           single_qub_gates_num=1,
                           two_qub_gates_num=0)
    ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)
    ideal_gates_list = list(pure_channels.values())
    eval_pure = QCEvaluator(ideal_gates_list, 1, dim)
    eval_pure.add_circuit(tn_template=simple_template(ncon_tmpls['0'], 1, ['H']), name='0')
    tomography_output = eval_pure.make_full_samples(name=str(0), bs_override=1000)
    tomography_res = compare_results(vanilla_res, tomography_output, err=0.05)
    true_res = {'0': 0.5, '1': 0.5}
    for idx, val in true_res.items():
        assert np.isclose(tomography_res[idx], val, atol=0.05)


def test_simple_2q_circuit():
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    CX = np.array([
        [[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
        [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]
    ], dtype=np.complex128)
    gate_dict = {
        "H": H,
        "CX": CX
    }
    dim = 2
    circuits = [["H_0", "CX_0_1"]]
    vanilla_res = simulate_circs(circuits, gate_dict, dim)

    pure_channels = {
        "H": c_util.convert_1qmatrix_to_channel(H),
        'CX': c_util.convert_2qmatrix_to_channel(CX, dim=dim)
    }
    gen = cg.DataGenerator(qubits_num=2,
                           gates_names=["H", "CX"],
                           single_qub_gates_num=1,
                           two_qub_gates_num=1)
    ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)
    ideal_gates_list = list(pure_channels.values())
    eval_pure = QCEvaluator(ideal_gates_list, 2, dim)
    eval_pure.add_circuit(tn_template=simple_template(ncon_tmpls['0'], 2, ['H']), name='0')
    tomography_output = eval_pure.make_full_samples(name=str(0), bs_override=1000)
    tomography_res = compare_results(vanilla_res, tomography_output, err=0.05)
    true_res = {'00': 0.5, '01': 0, '10': 0, '11': 0.5}
    for idx, val in true_res.items():
        assert np.isclose(tomography_res.get(idx, 0), val, atol=0.05)


def test_1q_random_circuit():
    for dim in [2, 3, 4]:
        U = unitary_group.rvs(dim)
        gate_dict = {
            "U": U,
        }
        circuits = [["U_0"]]
        vanilla_res = simulate_circs(circuits, gate_dict, dim)

        pure_channels = {
            "U": c_util.convert_1qmatrix_to_channel(U)
        }
        gen = cg.DataGenerator(qubits_num=1,
                               gates_names=["U"],
                               single_qub_gates_num=1,
                               two_qub_gates_num=0)
        ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)
        ideal_gates_list = list(pure_channels.values())
        eval_pure = QCEvaluator(ideal_gates_list, 1, dim)
        eval_pure.add_circuit(tn_template=ncon_tmpls['0'], name='0')
        tomography_output = eval_pure.make_full_samples(name=str(0), bs_override=1000)
        compare_results(vanilla_res, tomography_output, err=0.05)


def test_2qudit_random_circuit():
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
        vanilla_res = simulate_circs(circuits, gate_dict, dim)

        pure_channels = {
            "U1": c_util.convert_1qmatrix_to_channel(U1),
            "U2": c_util.convert_1qmatrix_to_channel(U2),
            "V": c_util.convert_2qmatrix_to_channel(V, dim)
        }
        gen = cg.DataGenerator(qubits_num=2,
                               gates_names=["U1", "U2", "V"],
                               single_qub_gates_num=2,
                               two_qub_gates_num=1)
        ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)
        ideal_gates_list = list(pure_channels.values())
        eval_pure = QCEvaluator(ideal_gates_list, 2, dim)
        eval_pure.add_circuit(tn_template=simple_template(ncon_tmpls['0'], 2, ['U1', 'U2']), name='0')
        tomography_output = eval_pure.make_full_samples(name=str(0), bs_override=1000)
        compare_results(vanilla_res, tomography_output, err=0.05)
