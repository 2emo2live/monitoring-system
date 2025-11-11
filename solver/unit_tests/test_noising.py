import pytest
import tensorflow as tf
import QGOpt as qgo
import typing as tp
import numpy as np

import solver.utils.general_utils as util
import solver.noising_tools as ns
import solver.utils.channel_utils as c_util
from solver.utils.misc import COMPLEX
from solver.utils.testing import same_matrix, is_choi, create_random_channel, is_dm

MANIF = qgo.manifolds.StiefelManifold()


# @pytest.mark.parametrize("d", [2])

def test_ad_kraus_construction():
    # Explict test for qubit Kraus operators
    for gamma in np.linspace(0, 1, 10):
        A0 = ns.create_ad_single_kraus(0, gamma, dim=2)
        A1 = ns.create_ad_single_kraus(1, gamma, dim=2)
        A0_check = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        A1_check = np.array([[0, np.sqrt(gamma)], [0, 0]])
        assert same_matrix(A0, A0_check)
        assert same_matrix(A1, A1_check)

    # Explicit test for qutrit Kraus operators
    for gamma in np.linspace(0, 1, 10):
        A0 = ns.create_ad_single_kraus(0, gamma, dim=3)
        A1 = ns.create_ad_single_kraus(1, gamma, dim=3)
        A2 = ns.create_ad_single_kraus(2, gamma, dim=3)

        A0_check = np.array([[1, 0, 0],
                             [0, np.sqrt(1 - gamma), 0],
                             [0, 0, 1 - gamma]])
        A1_check = np.array([[0, np.sqrt(gamma), 0],
                             [0, 0, np.sqrt(2 * gamma * (1 - gamma))],
                             [0, 0, 0]])
        A2_check = np.array([[0, 0, gamma],
                             [0, 0, 0],
                             [0, 0, 0]])
        assert same_matrix(A0, A0_check)
        assert same_matrix(A1, A1_check)
        assert same_matrix(A2, A2_check)

    # Check of normalization condition for Kraus operators
    for gamma in np.linspace(0, 1, 10):
        for d in [2, 3, 4]:
            sum_check = tf.zeros((d, d), dtype=COMPLEX)
            for i in range(d):
                A = ns.create_ad_single_kraus(i, gamma, dim=d)
                sum_check += tf.math.conj(tf.transpose(A) @ A)
            assert same_matrix(sum_check, tf.eye(d, dtype=COMPLEX))


def test_creating_Z():
    Z2_check = tf.constant([[1, 0], [0, -1]], dtype=COMPLEX)
    Z3_check = tf.constant([[1, 0, 0],
                            [0, np.exp(1j * 2 * np.pi / 3), 0],
                            [0, 0, np.exp(1j * 4 * np.pi / 3)]], dtype=COMPLEX)
    Z4_check = tf.constant([[1, 0, 0, 0],
                            [0, np.exp(1j * 2 * np.pi / 4), 0, 0],
                            [0, 0, np.exp(1j * 4 * np.pi / 4), 0],
                            [0, 0, 0, np.exp(1j * 6 * np.pi / 4)]], dtype=COMPLEX)

    assert same_matrix(ns.create_Z(2), Z2_check)
    assert same_matrix(ns.create_Z(3), Z3_check)
    assert same_matrix(ns.create_Z(4), Z4_check)

    assert same_matrix(ns.create_Z(2, 2), tf.tensordot(Z2_check, Z2_check, axes=1))
    assert same_matrix(ns.create_Z(3, 2), tf.tensordot(Z3_check, Z3_check, axes=1))
    assert same_matrix(ns.create_Z(4, 2), tf.tensordot(Z4_check, Z4_check, axes=1))


def test_pd_channel():
    # Testing zero noises
    for d in [2, 3, 4]:
        channel = ns.create_pd_channel(0.0, dim=d)
        assert same_matrix(channel, tf.eye(d ** 2, dtype=COMPLEX))
    # Testing phase damping
    for d in [2, 3, 4]:
        channel = ns.create_pd_channel(1.0, dim=d)

        # Making fully dephased non-normalized density matrix
        rho = tf.ones(d ** 2, dtype=COMPLEX)
        rho = tf.tensordot(channel, rho, axes=1)

        # Making fully dephased non-normalized density matrix for the test
        rho_check = tf.eye(d, dtype=COMPLEX)
        rho_check = tf.reshape(rho_check, (-1))
        rho_check = tf.constant(rho_check, dtype=COMPLEX)

        assert same_matrix(rho, rho_check)


def test_AP_channel():
    # Testing zero noises
    for d in [2, 3, 4]:
        channel = ns.create_AP_matrix(0.0, 0.0, dim=d)
        assert same_matrix(channel, tf.eye(d ** 2, dtype=COMPLEX))
    # Testing total amplitude damping
    for d in [2, 3, 4]:
        channel = ns.create_AP_matrix(1.0, 0.0, dim=d)

        # Making fully damped non-normalized density matrix
        rho = tf.eye(d, dtype=COMPLEX)
        rho = tf.reshape(rho, (-1))
        rho = tf.tensordot(channel, rho, axes=1)

        # Making fully damped non-normalized density matrix for the test
        rho_check = [0] * (d ** 2)
        rho_check[0] = d
        rho_check = tf.constant(rho_check, dtype=COMPLEX)

        assert same_matrix(rho, rho_check)


def test_depol_channel():
    # Testing zero noises
    for d in [2, 3, 4]:
        channel = ns.create_1q_depol_matrix(0.0, dim=d)
        assert same_matrix(channel, tf.eye(d ** 2, dtype=COMPLEX))
    for d in [2, 3, 4]:
        channel = ns.create_1q_depol_matrix(1.0, dim=d)
        rho1 = np.zeros((d, d), dtype=np.complex128)
        rho1[1, 1] = 1.0
        rho_in_vec = tf.reshape(rho1, [-1, 1])
        rho_out_vec = channel @ rho_in_vec
        rho_out = tf.reshape(rho_out_vec, [d, d])
        expected_rho_out = tf.eye(d, dtype=COMPLEX) / tf.cast(d, COMPLEX)
        assert same_matrix(rho_out, expected_rho_out)
    for d in [2, 3, 4]:
        for p in [0.25, 0.5, 0.7]:
            channel = ns.create_1q_depol_matrix(p, dim=d)
            states = [
                # Basis state |0⟩⟨0|
                tf.constant(np.diag([1] + [0] * (d - 1)), dtype=COMPLEX),
                # Basis state |1⟩⟨1| (if dim > 1)
                tf.constant(np.diag([0, 1] + [0] * (d - 2)), dtype=COMPLEX),
                # Maximally mixed state
                tf.eye(d, dtype=COMPLEX) / tf.cast(d, COMPLEX),
                # Coherent superposition
                tf.constant(np.ones((d, d), dtype=np.complex128) / d, dtype=COMPLEX)
            ]

            for rho in states:
                # Apply channel and check trace
                rho_vec = tf.reshape(rho, [-1, 1])
                rho_out_vec = channel @ rho_vec
                rho_out = tf.reshape(rho_out_vec, (d, d))

                trace_in = tf.linalg.trace(rho)
                trace_out = tf.linalg.trace(rho_out)

                assert abs(trace_in - trace_out) < 1e-10, (
                    f"Trace not preserved: dim={d}, p={p}, "
                    f"trace_in={trace_in.numpy()}, trace_out={trace_out.numpy()}"
                )



def test_zero_noise():
    channel = create_random_channel(1)
    assert same_matrix(ns.make_1q_hybrid_channel(channel, [0, 0, 0]), channel)

    channel2 = util.convert_2q_from16x16(create_random_channel(2))
    assert same_matrix(ns.make_2q_hybrid_channel(channel2, [0, 0, 0]), channel2)

    unitary1 = MANIF.random((2, 2), dtype=COMPLEX)
    unit_channel = c_util.convert_1qmatrix_to_channel(unitary1)
    assert same_matrix(ns.make_1q_4pars_channel(unit_channel, [0, 0, 0, 0]), unit_channel)

    unitary2 = MANIF.random((4, 4), dtype=COMPLEX)
    unit_channel2 = c_util.convert_2qmatrix_to_channel(util.convert_44_to_2222(unitary2))
    assert same_matrix(ns.make_2q_4pars_channel(unit_channel2, [0, 0, 0, 0]), unit_channel2)


def test_nkp():
    for _ in range(5):
        unitary1 = MANIF.random((2, 2), dtype=COMPLEX)
        unit_channel = c_util.convert_1qmatrix_to_channel(unitary1)
        extracted_unitary = ns.nearest_kron_product(unit_channel, 1)
        # assert same_matrix(extracted_unitary, unitary1) will NOT work because of arbitrary phase
        assert same_matrix(c_util.convert_1qmatrix_to_channel(extracted_unitary), unit_channel)

        unitary2 = MANIF.random((4, 4), dtype=COMPLEX)
        unit_channel2 = c_util.convert_2qmatrix_to_channel(util.convert_44_to_2222(unitary2))
        channel_16x16 = util.convert_2q_to16x16(unit_channel2)
        extracted_unitary2 = util.convert_44_to_2222(ns.nearest_kron_product(channel_16x16, 2))
        assert same_matrix(c_util.convert_2qmatrix_to_channel(extracted_unitary2), unit_channel2)


'''NOISE_PARAMS_1Q = [
    [[0.3, 0, 0, 0]],
    [[0, 0.3, 0, 0]],
    [[0, 0, 0.3, 0]],
    [[0, 0, 0, 0.3]],
    [[0.15, 0.2, 0.2, 0.5]],
    [[0.05, 0.1, 0.05, 0.1]],
]

NAMES_1Q = [
    'depol-0.3',
    'gamma1-0.3',
    'gamma2-0.3',
    'gauss-0.3',
    'combined1',
    'combined2'
]'''
NOISE_PARAMS_1Q = [
    [[0.3, 0, 0]],
    [[0, 0.3, 0]],
    [[0, 0, 0.3]],
    [[0.15, 0.2, 0.2]],
    [[0.05, 0.1, 0.05]],
]

NAMES_1Q = [
    'depol-0.3',
    'gamma1-0.3',
    'gamma2-0.3',
    'combined1',
    'combined2'
]


@pytest.mark.parametrize(['noise_list'], NOISE_PARAMS_1Q, ids=NAMES_1Q)
def test_1q_channel_noise_params(noise_list: list[int]):
    for _ in range(5):
        unitary = MANIF.random((2, 2), dtype=COMPLEX)
        channel = c_util.convert_1qmatrix_to_channel(unitary)
        noised_channel = ns.make_1q_4pars_channel(channel, noise_list)
        choi = util.choi_swap_1qchannel(noised_channel) / tf.constant(2, dtype=COMPLEX)
        assert is_choi(choi, eps=2e-6)
        rho_in = tf.transpose(tf.reshape(choi, (2, 2, 2, 2)), (0, 2, 1, 3))[:, :, 0, 0]
        assert is_dm(rho_in * 2, eps=2e-6) == 0


'''NOISE_PARAMS_2Q = [
    [[0.3, 0, 0, 0]],
    [[0, 0.3, 0, 0]],
    [[0, 0, 0.3, 0]],
    [[0, 0, 0, 0.3]],
    [[0.15, 0.2, 0.2, 0.5]],
    [[0.05, 0.1, 0.05, 0.1]],
]'''
NOISE_PARAMS_2Q = [
    [[0.3, 0, 0]],
    [[0, 0.3, 0]],
    [[0, 0, 0.3]],
    [[0.15, 0.2, 0.2]],
    [[0.05, 0.1, 0.05]],
]

@pytest.mark.parametrize(['noise_list'], NOISE_PARAMS_2Q, ids=NAMES_1Q)
def test_2q_channel_noise_params(noise_list: list[int]):
    for _ in range(5):
        unitary = util.convert_44_to_2222(MANIF.random((4, 4), dtype=COMPLEX))
        channel = c_util.convert_2qmatrix_to_channel(unitary)
        noised_channel = ns.make_2q_4pars_channel(channel, noise_list)
        choi = util.choi_swap_2qchannel(noised_channel) / tf.constant(4, dtype=COMPLEX)
        assert is_choi(choi, eps=2e-6)


'''NOISE_GRAD_TESTS = [
    (ns.make_1q_hybrid_channel, ns.make_2q_hybrid_channel, tf.Variable([0.1, 0.1, 0.1])),
    (ns.make_1q_4pars_channel, ns.make_2q_4pars_channel, tf.Variable([0.0, 0.0, 0.0, 0.3])),
    (ns.make_1q_4pars_channel, ns.make_2q_4pars_channel, tf.Variable([0.1, 0.1, 0.1, 0.1]))
]'''
NOISE_GRAD_TESTS = [
    (ns.make_1q_hybrid_channel, ns.make_2q_hybrid_channel, tf.Variable([0.1, 0.1, 0.1]))
]
#NAMES_GRAD = ['ad/pd/depol', 'gaussian-blur', 'combined']
NAMES_GRAD = ['ad/pd/depol']

@pytest.mark.parametrize(['func_1q', 'func_2q', 'params'], NOISE_GRAD_TESTS, ids=NAMES_GRAD)
def test_noise_does_not_kill_grad(func_1q: tp.Callable[[tf.Tensor, ...], tf.Tensor],
                                  func_2q: tp.Callable[[tf.Tensor, ...], tf.Tensor],
                                  params: tf.Variable):
    for _ in range(5):
        with tf.GradientTape() as tape:
            unitary = MANIF.random((2, 2), dtype=COMPLEX)
            channel = c_util.convert_1qmatrix_to_channel(unitary)
            noised_channel = func_1q(channel, params)
            loss = tf.math.abs(tf.linalg.norm(noised_channel - channel) ** 2) * 1000
            grad = tape.gradient(loss, params)

            for idx, elem in enumerate(grad):
                if tf.abs(params[idx]) > 1e-10:
                    assert elem > 1e-1

        with tf.GradientTape() as tape:
            unitary_2q = util.convert_44_to_2222(MANIF.random((4, 4), dtype=COMPLEX))
            channel_2q = c_util.convert_2qmatrix_to_channel(unitary_2q)
            noised_channel_2q = func_2q(channel_2q, params)
            loss = tf.math.abs(tf.linalg.norm(noised_channel_2q - channel_2q) ** 2) * 1000
            grad = tape.gradient(loss, params)

            for idx, elem in enumerate(grad):
                if tf.abs(params[idx]) > 1e-10:
                    assert elem > 1e-1

# TODO: more noised tests (maybe numeric assertions on create AD/PD matrix?)
