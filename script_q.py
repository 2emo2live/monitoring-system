import warnings
from cyclopts import App
import json

with warnings.catch_warnings():  # warning still somehow pierces through, so I cleared the cell output
    import tensorflow as tf  # tf 2.x
    import tensornetwork as tn

    tn.set_default_backend("tensorflow")

import numpy as np
import QGOpt as qgo
import solver.utils.general_utils as util
import solver.noising_tools as ns
import solver.utils.channel_utils as c_util
import solver.circuits_generation as cg
from solver.experiments import ExperimentConductor
from solver.utils.misc import INT, FLOAT, ID_GATE
from solver.QCSolver import get_complex_channel_form, QGOptSolverDebug


def prepare_channels(estim: dict[str, list[float]]) -> dict[str, tf.Tensor | tf.Variable]:
    params = {}
    for name in estim.keys():
        params[name] = qgo.manifolds.real_to_complex(tf.convert_to_tensor(estim[name], dtype=FLOAT))
        temp = []
        for i in range(len(estim[name])):
            temp.append(c_util.convert_channel_to_params(params[name][i]))
            temp[i] = tf.stack([tf.math.real(temp[i]), tf.math.imag(temp[i])], -1)
        params[name] = tf.Variable(tf.stack(temp, axis=0))
    return params


def apply_noise(noise: float, gates: dict[str, tf.Tensor], single_qub_gates: set[str],
                two_qub_gates: set[str]) -> dict[str, tf.Tensor]:
    noisy_gates = {}
    init_noise = tf.convert_to_tensor([noise, 0.0, 0.0], dtype=FLOAT)
    for name in gates:
        if name in single_qub_gates:
            noised_channel = ns.make_1q_hybrid_channel(gates[name], init_noise)
            params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(noised_channel))
            noisy_gates[name] = tf.Variable(tf.concat([params[tf.newaxis]], axis=0))
        elif name in two_qub_gates:
            noised_channel = ns.make_2q_hybrid_channel(gates[name], init_noise)
            params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(noised_channel))
            noisy_gates[name] = tf.Variable(tf.concat([params[tf.newaxis]]), axis=0)
        elif name == ID_GATE:
            params = qgo.manifolds.complex_to_real(c_util.convert_channel_to_params(gates[ID_GATE]))
            noisy_gates[ID_GATE] = tf.Variable(params[tf.newaxis])
    return gates


def form_output(estimated_gates_dict: dict[str, tf.Tensor]) -> dict[str, list[float]]:
    dict_result = get_complex_channel_form(estimated_gates_dict)
    for name in dict_result.keys():
        dict_result[name] = tf.stack([tf.math.real(dict_result[name]), tf.math.imag(dict_result[name])],
                                     -1).numpy().tolist()
    return dict_result


def prepare_result(res: list[dict[str, int]]) -> list[np.ndarray]:
    f_res = []
    for i in range(len(res)):
        single_res = []
        for j in res[i].keys():
            for k in range(res[i][j]):
                single_res.append(list(map(int, list(j))))
        f_res.append(np.array(single_res))
    return f_res


def compute(circuits: list[list[str]], results: list[dict[str, int]], estimates_input: dict[str, list[float]],
            iter_num: int,
            lr: float, lmbd1: int, lmbd2: int, noise: float) -> dict[str, list[float]]:
    NUM_QUBITS = len(list(results[0].keys())[0])
    gates_labels = list(estimates_input.keys())
    gates_labels.remove('_E')
    single_qub_gates = set()
    two_qub_gates = set()
    estimates = prepare_channels(estimates_input)
    for i in gates_labels:
        if estimates[i].shape[1] == 4:
            single_qub_gates.add(i)
        else:
            two_qub_gates.add(i)
    gen = cg.DataGenerator(qubits_num=NUM_QUBITS,
                           gates_names=gates_labels,
                           single_qub_gates_num=len(single_qub_gates),
                           two_qub_gates_num=len(two_qub_gates))
    ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)

    noise_cfg = []
    with warnings.catch_warnings():
        exp_test = ExperimentConductor(pure_channels_set=estimates,
                                       noise_cfg=noise_cfg,
                                       exp_name='zh_test',
                                       qubits_num=NUM_QUBITS,
                                       lr=lr,
                                       lmbd1=lmbd1,
                                       lmbd2=lmbd2,
                                       iterations=iter_num)

    QC_t = QGOptSolverDebug(qubits_num=NUM_QUBITS,
                            single_qub_gates_names=single_qub_gates,
                            two_qub_gates_names=two_qub_gates,
                            pure_channels_set=get_complex_channel_form(estimates),
                            compress_samples=True,
                            noise_params=exp_test.noise_params,
                            noise_iter0=noise,
                            initial_estimated_gates_override=estimates)

    if noise == 0:
        QC_t.estimated_gates_dict = estimates
    else:
        QC_t.estimated_gates_dict = apply_noise(noise, estimates, single_qub_gates, two_qub_gates)

    for name, tmpl in ncon_tmpls.items():
        QC_t.add_circuit(tn_template=tmpl, name=name)

    format_res = prepare_result(results)
    QC_t.samples_compressed = {}
    for idx, outcome in enumerate(format_res):
        dimdim = tf.constant([2] * NUM_QUBITS, dtype=INT)
        ids = util.ravel_multi_index(tf.convert_to_tensor(outcome), dimdim)
        compressor = np.bincount(ids, minlength=2 ** NUM_QUBITS)
        QC_t.samples_compressed[str(idx)] = tf.convert_to_tensor(compressor, dtype=FLOAT)

    manif = qgo.manifolds.ChoiMatrix()
    opt_t = qgo.optimizers.RAdam(manif, exp_test.lr)

    loss_dynamics_t, norms_dict, _ = QC_t.train_optimizer(opt=opt_t,
                                                          lmbd1=exp_test.lmbd1,
                                                          lmbd2=exp_test.lmbd2,
                                                          iters=exp_test.iters,
                                                          v=1,
                                                          fid_ctr=-1,
                                                          norm_ctr=-1)

    return form_output(QC_t.estimated_gates_dict)


app = App()


@app.default
def default():                              # for testing
    with open('input_example/config.json', 'r') as file:
        config = json.load(file)
    with open('input_example/circs.json', 'r') as file:
        circs = json.load(file)
    with open('input_example/results.json', 'r') as file:
        results = json.load(file)
    with open('input_example/start_estimations.json', 'r') as file:
        estimates = json.load(file)

    result_dict = compute(circs, results, estimates, config['iters'], config['lr'],
                          config['lmbd1'], config['lmbd2'], config['noise'])

    json_result = json.dumps(result_dict)
    with open("estimations.json", 'w') as file:
        file.write(json_result)


@app.command
def run(config_file: str, circs_file: str, res_file: str, estim_file: str):
    with open(config_file, 'r') as file:
        config = json.load(file)
    with open(circs_file, 'r') as file:
        circs = json.load(file)
    with open(res_file, 'r') as file:
        results = json.load(file)
    with open(estim_file, 'r') as file:
        estimates = json.load(file)

    result_dict = compute(circs, results, estimates, config['iters'], config['lr'],
                          config['lmbd1'], config['lmbd2'], config['noise'])

    json_result = json.dumps(result_dict)
    with open("estimations.json", 'w') as file:
        file.write(json_result)


app()
