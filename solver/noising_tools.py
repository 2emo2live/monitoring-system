import tensorflow as tf
import numpy as np
from math import factorial as fact

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
from solver.utils.misc import COMPLEX, TENSOR

import sys


def create_Z(dim: int = 2, pow: int = 1):
    Z = np.eye(dim, dtype=np.complex128)
    omega = np.exp(pow * 1j * 2 * np.pi / dim)
    coef = 1
    for i in range(dim):
        Z[i][i] = coef
        coef *= omega
    return tf.constant(Z, dtype=COMPLEX)


'''def create_X(dim: int = 2):
    X = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        X[i][(i + 1) % dim] = 1
    return tf.constant(X, dtype=COMPLEX)'''


def create_X(dim: int = 2, pow: int = 1):
    X = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        X[i][(i + pow) % dim] = 1
    return tf.constant(X, dtype=COMPLEX)


@tf.function
def create_1q_depol_matrix(p: TENSOR, dim: int = 2) -> TENSOR:
    """
    Creates a Tensor(4, 4)[complex128] describing a 1-qudit depolarizing quantum channel
    """
    E = tf.eye(dim, dtype=COMPLEX)
    E_channel = c_util.convert_1qmatrix_to_channel(E)

    pauli_basis = []
    for a in range(dim):
        for b in range(dim):
            X_a = create_X(dim, a)
            Z_b = create_Z(dim, b)
            #phase = np.exp(1j * np.pi * a * b / dim)
            #pauli = tf.cast(phase, COMPLEX) * (X_a @ Z_b)
            pauli = (X_a @ Z_b)
            pauli_basis.append(pauli)

    num_paulis = dim * dim

    depol = E_channel * (1 - p)

    for pauli in pauli_basis:
        pauli_channel = c_util.convert_1qmatrix_to_channel(pauli)
        depol += pauli_channel * (p / tf.cast(num_paulis, COMPLEX))

    return depol


'''@tf.function
def create_2q_depol_matrix(p: TENSOR, dim: int = 2):
    """
    Creates a Tensor(4, 4)[complex128] describing a 2-qubit depolarizing quantum channel
    """
    E = tf.eye(dim, dtype=COMPLEX)
    E_channel = c_util.convert_1qmatrix_to_channel(E)
    big_e_channel = util.kron(E_channel, E_channel)
    depol = big_e_channel * (1 - p)

    pauli_basis = []
    for a in range(dim):
        for b in range(dim):
            X_a = create_X(dim, a)
            Z_b = create_Z(dim, b)
            pauli = (X_a @ Z_b)
            pauli_basis.append(pauli)

    for m1 in pauli_basis:
        for m2 in pauli_basis:
            m = np.kron(m1, m2)
            m = util.swap_legs(tf.reshape(m, (dim, dim, dim, dim)))
            depol += c_util.convert_2qmatrix_to_channel(m) * p * (1 / dim**4)
    return depol'''


#@tf.function
def create_AP_matrix(gamma1: TENSOR, gamma2: TENSOR, dim: int = 2):
    """
    Args:
        gamma1: Tensor()[float] - parameter for amplitude damping
        gamma2: Tensor()[float] - parameter for phase damping

    Returns:
        Tensor(4, 4)[complex128] describing a 1-qudit amplitude damping & phase damping quantum channel
    """
    channel = tf.zeros((dim ** 2, dim ** 2), dtype=COMPLEX)
    for i in range(dim):
        ad_kraus = create_ad_single_kraus(i, gamma1, dim=dim)
        channel += c_util.convert_1qmatrix_to_channel(ad_kraus)

    pd_channel = create_pd_channel(gamma2, dim)
    channel = channel @ pd_channel

    return channel


def create_ad_single_kraus(k: int, gamma: TENSOR, dim: int = 2):
    """
    Generates k-th amplitude damping Kraus operator
    """
    kraus = tf.zeros((dim, dim), dtype=COMPLEX)
    for r in range(k, dim):
        basis_element = util.make_basis_state_matrix(r - k, r, dim=dim)
        coef = tf.cast(tf.math.sqrt(fact(r) / fact(k) / fact(r - k)), dtype=tf.float64) * \
               tf.cast(tf.math.sqrt((1 - gamma) ** (r - k) * gamma ** k), dtype=tf.float64)
        coef = tf.cast(coef, dtype=COMPLEX)
        kraus += coef * basis_element
    return kraus


def create_pd_channel(gamma: TENSOR, dim: int = 2):
    p = tf.cast((dim - 1) / dim * gamma, dtype=COMPLEX)  # phase flip probability
    channel = c_util.convert_1qmatrix_to_channel(tf.math.sqrt(1 - p) * tf.eye(dim, dtype=COMPLEX))
    for i in range(dim - 1):
        kraus = tf.math.sqrt(p / (dim - 1)) * create_Z(dim, pow=i + 1)
        channel += c_util.convert_1qmatrix_to_channel(kraus)
    return channel


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
    gamma1 = args_list[1] / 2
    gamma2 = args_list[2] / 2
    ap_channel = create_AP_matrix(gamma1, gamma2, dim)
    dp_channel = create_1q_depol_matrix(p, dim)

    # TODO: check correctness
    output = dp_channel @ target
    output = ap_channel @ output @ ap_channel

    return output


@tf.function
def make_2q_hybrid_channel(target: TENSOR, args_list: TENSOR, dim: int = 2) -> TENSOR:
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

    ap_channel = create_AP_matrix(gamma1, gamma2, dim)
    ap_channel_2q = util.kron(ap_channel, ap_channel)

    dp_channel = create_1q_depol_matrix(p, dim)
    dp_channel_2q = util.kron(dp_channel, dp_channel)

    # TODO: check correctness
    reshaped_target = tf.reshape(target, (dim**4, dim**4))
    output = ap_channel_2q @ dp_channel_2q @ reshaped_target @ ap_channel_2q
    output = tf.reshape(output, (dim**2, dim**2, dim**2, dim**2))

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


'''@tf.function
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

    left_matrix = util.kron(eigenvecs, tf.math.conj(eigenvecs), dim)

    right_matrix = util.kron(tf.linalg.adjoint(eigenvecs), tf.transpose(eigenvecs), dim)

    return left_matrix @ middle_matrix @ right_matrix


@tf.function
def create_2q_dispersed_channel(target: TENSOR, sigma: float, dim: int = 2) -> TENSOR:
    """
    TODO: Write docstring
    """
    basic_gate = nearest_kron_product(util.convert_2q_to16x16(target, dim), 2, dim)

    eigenvals, eigenvecs = tf.linalg.eig(basic_gate)
    lambds = tf.math.log(eigenvals) * -1j
    new_eigenvals = _pseudokron_eigs(lambds)
    middle_matrix = tf.linalg.diag(_dispersed_eigs(new_eigenvals, sigma))

    left_matrix = util.kron(eigenvecs, tf.math.conj(eigenvecs))
    right_matrix = util.kron(tf.linalg.adjoint(eigenvecs), tf.transpose(eigenvecs))
    wrong_shaped_matrix = left_matrix @ middle_matrix @ right_matrix
    good_matrix = util.convert_2q_from16x16(wrong_shaped_matrix, dim)

    return good_matrix'''


@tf.function
def make_1q_4pars_channel(target: TENSOR, args_list: list[float], dim: int = 2) -> TENSOR:
    """
    TODO: Write docstring
    """
    # assert (len(args_list) == 4)
    # p_dep, gamma1, gamma2, sigma = args_list

    # disp_channel = create_1q_dispersed_channel(target, args_list[0], dim)
    # output = make_1q_hybrid_channel(disp_channel, args_list[1:], dim, ind)
    output = make_1q_hybrid_channel(target, args_list, dim)

    return output


@tf.function
def make_2q_4pars_channel(target: TENSOR, args_list: list[float], dim: int = 2) -> TENSOR:
    """
    TODO: Write docstring
    """
    # assert (len(args_list) == 4)
    # p_dep, gamma1, gamma2, sigma = args_list
    # disp_channel = create_2q_dispersed_channel(target, args_list[0], dim)
    # output = make_2q_hybrid_channel(disp_channel, args_list[1:], dim)
    output = make_2q_hybrid_channel(target, args_list, dim)

    return output
