import tensorflow as tf
import numpy as np

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
from solver.utils.misc import COMPLEX, TENSOR

import sys


def create_Z(dim: int = 2):
    Z = np.eye(dim, dtype=np.complex128)
    omega = np.exp(1j * 2 * np.pi / dim)
    coef = 1
    for i in range(dim):
        Z[i][i] = coef
        coef *= omega
    return Z


def create_X(dim: int = 2):
    X = np.eye(dim - 1, dtype=np.complex128)
    X = np.concatenate([X, np.zeros((dim - 1, 1), dtype=np.complex128)], axis=1)
    new_row = np.zeros((1, dim), dtype=np.complex128)
    new_row[0][-1] = 1
    X = np.concatenate([X, new_row], axis=0)

    return X


def create_sigmaX(dim: int = 2, ind: int = 0) -> TENSOR:
    # TODO: add other indices
    assert (ind == 0, "WIP")
    ket_0 = np.zeros(dim, dtype=np.complex128)
    ket_0[0] = 1
    ket_1 = np.zeros(dim, dtype=np.complex128)
    ket_1[1] = 1
    ketbra_01 = np.tensordot(ket_0, ket_1.T, axes=0)
    ketbra_10 = np.tensordot(ket_1, ket_0.T, axes=0)
    sigmaX = ketbra_01 - ketbra_10
    return tf.convert_to_tensor(sigmaX, dtype=COMPLEX)


def create_sigmaY(dim: int = 2, ind: int = 0) -> TENSOR:
    assert (ind == 0, "WIP")
    ket_0 = np.zeros(dim, dtype=np.complex128)
    ket_0[0] = 1
    ket_1 = np.zeros(dim, dtype=np.complex128)
    ket_1[1] = 1
    ketbra_01 = np.tensordot(ket_0, ket_1.T, axes=0)
    ketbra_10 = np.tensordot(ket_1, ket_0.T, axes=0)
    sigmaY = ketbra_01 * 1j - ketbra_10 * 1j
    return tf.convert_to_tensor(sigmaY, dtype=COMPLEX)


def create_sigmaZ(dim: int = 2, ind: int = 0) -> TENSOR:
    assert (ind == 0, "WIP")
    ket_0 = np.zeros(dim, dtype=np.complex128)
    ket_0[0] = 1
    ket_1 = np.zeros(dim, dtype=np.complex128)
    ket_1[1] = 1
    ketbra_00 = np.tensordot(ket_0, ket_0.T, axes=0)
    ketbra_11 = np.tensordot(ket_1, ket_1.T, axes=0)
    sigmaZ = ketbra_00 - ketbra_11
    return tf.convert_to_tensor(sigmaZ, dtype=COMPLEX)


# E = tf.eye(2, dtype=COMPLEX)
# E_channel = c_util.convert_1qmatrix_to_channel(E)


@tf.function
def create_1q_depol_matrix(p: TENSOR, dim: int = 2) -> TENSOR:
    """
    Creates a Tensor(4, 4)[complex128] describing a 1-qubit depolarizing quantum channel
    """
    E = tf.eye(dim, dtype=COMPLEX)
    e1_channel = c_util.convert_1qmatrix_to_channel((1 - p) * E)
    Z = create_Z(dim)
    X = create_X(dim)
    e2 = np.zeros((dim, dim), dtype=np.complex128)
    temp_z = np.eye(dim, dtype=np.complex128)
    for i in range(dim):
        temp_x = np.eye(dim, dtype=np.complex128)
        for j in range(dim):
            if (i == 0) and (j == 0):
                continue
            e2 += temp_x @ temp_z
            temp_x = temp_x @ X
        temp_z = temp_z @ Z
    e2 *= tf.math.sqrt(p/(dim*dim - 1))
    e2_channel = c_util.convert_1qmatrix_to_channel(tf.convert_to_tensor(e2))

    return e1_channel + e2_channel


@tf.function
def create_2q_depol_matrix(p: TENSOR, dim: int = 2):
    """
    Creates a Tensor(4, 4)[complex128] describing a 2-qubit depolarizing quantum channel
    """
    E = tf.eye(dim, dtype=COMPLEX)
    E_channel = c_util.convert_1qmatrix_to_channel(E)
    big_e_channel = util.kron(E_channel, E_channel)
    depol = big_e_channel * (1 - p)
    sigmaX = create_sigmaX(1, 1)
    sigmaY = create_sigmaY(1, 1)
    sigmaZ = create_sigmaZ(1, 1)
    for m1 in [sigmaX, sigmaY, sigmaZ, E]:
        for m2 in [sigmaX, sigmaY, sigmaZ, E]:
            m = np.kron(m1, m2)
            m = util.swap_legs(tf.reshape(m, (2, 2, 2, 2)))
            depol += c_util.convert_2qmatrix_to_channel(m) * p * 0.0625  # 1/16
    return depol


@tf.function
def create_AP_matrix(gamma: TENSOR, dim: int = 2):
    """
    Args:
        gamma1: Tensor()[float] - parameter for phase damping

    Returns:
        Tensor(4, 4)[complex128] describing a 1-qudit amplitude damping & phase damping quantum channel
    """
    E = np.eye(dim, dtype=np.complex128)
    Z = create_Z(dim)
    e1_channel = c_util.convert_1qmatrix_to_channel(tf.cast(tf.math.sqrt(1 - gamma / 2), dtype=COMPLEX) * tf.convert_to_tensor(E, dtype=COMPLEX))
    e2_channel = c_util.convert_1qmatrix_to_channel(tf.cast(tf.math.sqrt(gamma / 2), dtype=COMPLEX) * tf.convert_to_tensor(Z, dtype=COMPLEX))

    return e1_channel + e2_channel


@tf.function
def make_1q_hybrid_channel(target: TENSOR, args_list: TENSOR, dim: int = 2) -> TENSOR:
    """
    Args:
        target: a Tensor(4,4)[complex128] - a channel, which we are noising now will be applied
        args_list: a Tensor(3)[float] containing arguments for applying noise models.
        First arg is p for depolarization, and args 2 & 3 are for gamma1 & gamma2 - params for APD
        dim: number of dimensions of qudit

    Returns:
        Tensor(4,4)[complex128] - new noised channel
    """
    p = tf.cast(args_list[0], COMPLEX)
    gamma = args_list[1] / 2

    ap_channel = create_AP_matrix(gamma, dim)
    dp_channel = create_1q_depol_matrix(p, dim)

    #TODO: check correctness
    output = ap_channel @ dp_channel @ target @ ap_channel

    return output


@tf.function
def make_2q_hybrid_channel(target: TENSOR, args_list: TENSOR) -> TENSOR:
    """
    Args:
        target: a Tensor(4,4)[complex128] - a channel, which we are noising now will be applied
        args_list: a Tensor(3)[float] containing arguments for applying noise models.
        First arg is p for depolarization, and args 2 & 3 are for gamma1 & gamma2 - params for APD
    Returns:
        Tensor(4,4)[complex128] - new noised channel
    """
    p = tf.cast(args_list[0], COMPLEX)
    gamma1 = args_list[1] / 2
    gamma2 = args_list[2] / 2

    ap_channel = create_AP_matrix(gamma1, gamma2)
    ap_channel_2q = util.kron(ap_channel, ap_channel)

    dp_channel = create_1q_depol_matrix(p)
    dp_channel_2q = util.kron(dp_channel, dp_channel)

    # TODO: check correctness
    reshaped_target = tf.reshape(target, (16, 16))
    output = ap_channel_2q @ dp_channel_2q @ reshaped_target @ ap_channel_2q
    output = tf.reshape(output, (4, 4, 4, 4))

    return output


@tf.function
def nearest_kron_product(A: TENSOR, n_qd: int, dim: int = 2) -> TENSOR:
    """
    Yields nearest Kronecker product to a matrix.

    Given a matrix A and a shape, solves the problem
    min || A - kron(B, C) ||_{Fro}^2
    where the minimization is over B with (the specified shape) and C.
    The size of the SVD computed in this implementation is the size of the input
    argument A, and so the complexity scales like O((N^2)^3) = O(N^6).
    Args:
        A: m x n matrix
        n_qd: number of qudits which define the dimension
        dim: number of dimensions of qudits
    Returns:
        Approximating factor B (but calculates both B and C)
    """
    Bshape = [dim ** n_qd, dim ** n_qd]
    # Cshape = A.shape[0] // Bshape[0], A.shape[1] // Bshape[1]

    blocks = map(lambda blockcol: tf.split(blockcol, Bshape[0], 0),
                 tf.split(A, Bshape[1], 1))
    Atilde = tf.stack([tf.reshape(block, (-1,)) for blockcol in blocks
                       for block in blockcol])

    s, U, V = tf.linalg.svd(Atilde)
    idx = tf.argmax(s)
    s = tf.cast(s, COMPLEX)
    U = tf.cast(U, COMPLEX)
    V = tf.cast(V, COMPLEX)

    B = tf.math.sqrt(s[idx]) * tf.transpose(tf.reshape(U[:, idx], Bshape))
    # C = np.sqrt(s[idx]) * V[idx, :].reshape(Cshape)

    return tf.convert_to_tensor(B, dtype=COMPLEX)


@tf.function
def _pseudokron_eigs(lambds: TENSOR) -> TENSOR:
    """
    TODO: Write docstring
    """
    return tf.reshape(tf.transpose(tf.stack([lambds] * len(lambds))) - lambds, (-1,))


@tf.function
def _dispersed_eigs(lambds: TENSOR, sigma: float) -> TENSOR:
    """
    TODO: Write docstring
    """
    sigma = tf.cast(sigma, COMPLEX)
    return tf.math.exp(1j * lambds - lambds ** 2 * sigma ** 2 / 2)


@tf.function
def create_1q_dispersed_channel(target: TENSOR, sigma: float, dim: int = 2) -> TENSOR:
    """
    TODO: Write docstring
    """
    # it seems that Choi form coincides with our channel in single-qubit case
    basic_gate = nearest_kron_product(target, 1, dim)

    eigenvals, eigenvecs = tf.linalg.eig(basic_gate)

    lambds = tf.math.log(eigenvals) * -1j

    new_eigenvals = _pseudokron_eigs(lambds)
    middle_matrix = tf.linalg.diag(_dispersed_eigs(new_eigenvals, sigma))

    left_matrix = util.kron(eigenvecs, tf.math.conj(eigenvecs))

    right_matrix = util.kron(tf.linalg.adjoint(eigenvecs), tf.transpose(eigenvecs))

    return left_matrix @ middle_matrix @ right_matrix


@tf.function
def create_2q_dispersed_channel(target: TENSOR, sigma: float) -> TENSOR:
    """
    TODO: Write docstring
    """
    basic_gate = nearest_kron_product(util.convert_2q_to16x16(target), 2)

    eigenvals, eigenvecs = tf.linalg.eig(basic_gate)
    lambds = tf.math.log(eigenvals) * -1j
    new_eigenvals = _pseudokron_eigs(lambds)
    middle_matrix = tf.linalg.diag(_dispersed_eigs(new_eigenvals, sigma))

    left_matrix = util.kron(eigenvecs, tf.math.conj(eigenvecs))
    right_matrix = util.kron(tf.linalg.adjoint(eigenvecs), tf.transpose(eigenvecs))
    wrong_shaped_matrix = left_matrix @ middle_matrix @ right_matrix
    good_matrix = util.convert_2q_from16x16(wrong_shaped_matrix)

    return good_matrix


@tf.function
def make_1q_4pars_channel(target: TENSOR, args_list: list[float], dim: int = 2) -> TENSOR:
    """
    TODO: Write docstring
    """
    # assert (len(args_list) == 4)
    # p_dep, gamma1, gamma2, sigma = args_list

    #disp_channel = create_1q_dispersed_channel(target, args_list[0], dim)
    #output = make_1q_hybrid_channel(disp_channel, args_list[1:], dim, ind)
    output = make_1q_hybrid_channel(target, args_list, dim)

    return output


@tf.function
def make_2q_4pars_channel(target: TENSOR, args_list: list[float], dim: int = 2, ind: int = 0) -> TENSOR:
    """
    TODO: Write docstring
    """
    # assert (len(args_list) == 4)
    # p_dep, gamma1, gamma2, sigma = args_list
    disp_channel = create_2q_dispersed_channel(target, args_list[0])
    output = make_2q_hybrid_channel(disp_channel, args_list[1:])

    return output
