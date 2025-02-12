import QGOpt as qgo
import pytest
import tensorflow as tf

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
from solver.utils.misc import COMPLEX
from solver.utils.testing import same_matrix, is_choi, create_random_channel


MANIF = qgo.manifolds.StiefelManifold()

# All tests have been adopted to quDits!

def test_1q_channel_creation():
    """
    Test construction of channels from 1-quDit unitaries
    """
    for d in range(2, 5):
        for _ in range(5):
            unitary = MANIF.random((d, d), dtype=COMPLEX)
            human_channel = util.kron(unitary, tf.math.conj(unitary))
            assert same_matrix(human_channel, c_util.convert_1qmatrix_to_channel(unitary))


def test_1q_channel_choi_sanity_check():
    """
    This one does not check the Choi conversion itself (it is in test_utils)
    but rather if the created channel satisfies some obvious conditions
    """
    for d in range(2, 5):
        for _ in range(5):
            unitary = MANIF.random((d, d), dtype=COMPLEX)
            channel = c_util.convert_1qmatrix_to_channel(unitary)
            choi = util.choi_swap_1qchannel(channel, dim=d) / tf.constant(d, dtype=COMPLEX)
            assert is_choi(choi, dim=d), f'd = {d}'  # testing with explict dimensions in args
            assert is_choi(choi), f'd = {d}'  # testing without explicit dimensions in args


def test_2q_channel_choi_sanity_check():
    """
    Makes a test for 2-quDit Choi matrices
    """
    for d in range(2, 5):
        for _ in range(5):
            unitary2222 = util.convert_44_to_2222(MANIF.random((d**2, d**2), dtype=COMPLEX), dim=d)
            channel = c_util.convert_2qmatrix_to_channel(unitary2222, dim=d)
            choi = util.choi_swap_2qchannel(channel, dim=d) / tf.constant(d**2, dtype=COMPLEX)
            assert is_choi(choi)  # testing with explict dimensions in args
            assert is_choi(choi, dim=d**2)  # testing without explicit dimensions in args


def test_2q_channel_conversion():
    for d in range(2, 5):
        for _ in range(5):
            unitary = MANIF.random((d**2, d**2), dtype=COMPLEX)
            unitary_2222 = util.convert_44_to_2222(unitary, dim=d)
            ncon_channel = c_util.convert_2qmatrix_to_channel(unitary_2222, dim=d)

            human_channel = util.kron(unitary, tf.math.conj(unitary))

            assert same_matrix(human_channel, util.convert_2q_to16x16(ncon_channel, dim=d))
            assert same_matrix(ncon_channel, util.convert_2q_from16x16(human_channel, dim=d))

            assert same_matrix(human_channel,
                               util.convert_2q_to16x16(util.convert_2q_from16x16(human_channel, dim=d), dim=d))
            assert same_matrix(ncon_channel,
                               util.convert_2q_from16x16(util.convert_2q_to16x16(ncon_channel, dim=d), dim=d))


def test_qgo_conversion():
    for d in range(2, 5):
        for _ in range(5):
            ch1 = create_random_channel(1, d=d)
            pars1 = c_util.convert_channel_to_params(ch1, dim=d)
            ch1_twin = c_util.convert_params_to_channel(pars1, dim=d)
            assert same_matrix(ch1, ch1_twin)

            ch2 = util.convert_2q_from16x16(create_random_channel(2, d=d), dim=d)
            pars2 = c_util.convert_channel_to_params(ch2, dim=d)
            ch2_twin = c_util.convert_params_to_channel(pars2, dim=d)
            assert same_matrix(ch2, ch2_twin)
