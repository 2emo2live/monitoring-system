import tensorflow as tf
import numpy as np

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
from solver.utils.misc import COMPLEX, TENSOR

import sys

def create_sigmaX(dim: int = 2, ind: int = 0) -> TENSOR:
    #TODO: add other indices
    assert(ind == 0, "WIP")
    ket_0 = np.zeros(dim, dtype=np.complex128)
    ket_0[0] = 1
    ket_1 = np.zeros(dim, dtype=np.complex128)
    ket_1[1] = 1
    ketbra_01 = np.tensordot(ket_0, ket_1.T, axes=0)
    ketbra_10 = np.tensordot(ket_1, ket_0.T, axes=0)
    sigmaX = ketbra_01 - ketbra_10
    return tf.convert_to_tensor(sigmaX, dtype=COMPLEX)


def create_sigmaY(dim: int = 2, ind: int = 0) -> TENSOR:
    assert(ind == 0, "WIP")
    ket_0 = np.zeros(dim, dtype=np.complex128)
    ket_0[0] = 1
    ket_1 = np.zeros(dim, dtype=np.complex128)
    ket_1[1] = 1
    ketbra_01 = np.tensordot(ket_0, ket_1.T, axes=0)
    ketbra_10 = np.tensordot(ket_1, ket_0.T, axes=0)
    sigmaY = ketbra_01*1j - ketbra_10*1j
    return tf.convert_to_tensor(sigmaY, dtype=COMPLEX)


def create_sigmaZ(dim: int = 2, ind: int = 0) -> TENSOR:
    assert(ind == 0, "WIP")
    ket_0 = np.zeros(dim, dtype=np.complex128)
    ket_0[0] = 1
    ket_1 = np.zeros(dim, dtype=np.complex128)
    ket_1[1] = 1
    ketbra_00 = np.tensordot(ket_0, ket_0.T, axes=0)
    ketbra_11 = np.tensordot(ket_1, ket_1.T, axes=0)
    sigmaZ = ketbra_00 - ketbra_11
    return tf.convert_to_tensor(sigmaZ, dtype=COMPLEX)


#E = tf.eye(2, dtype=COMPLEX)
#E_channel = c_util.convert_1qmatrix_to_channel(E)


@tf.function
def create_1q_depol_matrix(p: TENSOR, dim: int = 2, ind: int = 0) -> TENSOR:
    """
    Creates a Tensor(4, 4)[complex128] describing a 1-qubit depolarizing quantum channel
    """
    E = tf.eye(dim, dtype=COMPLEX)
    E_channel = c_util.convert_1qmatrix_to_channel(E)
    depol = (c_util.convert_1qmatrix_to_channel(create_sigmaX(dim, ind)) * p * 0.25 +
             c_util.convert_1qmatrix_to_channel(create_sigmaY(dim, ind)) * p * 0.25 +
             c_util.convert_1qmatrix_to_channel(create_sigmaZ(dim, ind)) * p * 0.25 +
             E_channel * (1 - 0.75 * p))
    return depol


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
def create_AP_matrix(gamma1: TENSOR, gamma2: TENSOR, dim: int = 2, ind: int = 0):
    """
    Args:
        gamma1: Tensor()[float] - parameter for amplitude damping
        gamma2: Tensor()[float] - parameter for phase damping

    Returns:
        Tensor(4, 4)[complex128] describing a 1-qudit amplitude damping & phase damping quantum channel
    """

    num_of_q = int(np.floor(np.log2(dim)))
    E = np.eye(4, dtype=np.cdouble)
    E = tf.convert_to_tensor(E, dtype=COMPLEX)

    e0_a = tf.convert_to_tensor([[1, 0], [0, tf.math.sqrt(1 - gamma1)]], dtype=COMPLEX)
    e0_p = tf.convert_to_tensor([[1, 0], [0, tf.math.sqrt(1 - gamma2)]], dtype=COMPLEX)
    e1_p = tf.convert_to_tensor([[0, 0], [0, tf.math.sqrt(gamma2)]], dtype=COMPLEX)
    e1_a = tf.convert_to_tensor([[0, tf.math.sqrt(gamma1)], [0, 0]], dtype=COMPLEX)
    e0_a_channel = c_util.convert_1qmatrix_to_channel(e0_a)
    e0_p_channel = c_util.convert_1qmatrix_to_channel(e0_p)
    e1_p_channel = c_util.convert_1qmatrix_to_channel(e1_p)
    e1_a_channel = c_util.convert_1qmatrix_to_channel(e1_a)

    ap_channel = (e0_a_channel + e1_a_channel) @ (e0_p_channel + e1_p_channel)

    if ind != 0:
        E_n = E
        for i in range(1, ind):
            E_n = util.kron(E_n, E)
        ap_channel = util.kron(E_n, ap_channel)

    for i in range(ind + 1, num_of_q):
        ap_channel = util.kron(ap_channel, E)

    ap_channel = tf.reshape(ap_channel, (2**(2*num_of_q), 2**(2*num_of_q)))

    '''e0_a = np.array([[1, 0], [0, np.sqrt(1 - gamma1)]], dtype=np.cdouble)
    e0_p = np.array([[1, 0], [0, np.sqrt(1 - gamma2)]], dtype=np.cdouble)
    e1_p = np.array([[0, 0], [0, np.sqrt(gamma2)]], dtype=np.cdouble)
    e1_a = np.array([[0, np.sqrt(gamma1)], [0, 0]], dtype=np.cdouble)

    if ind != 0:
        for i in range(1, ind):
            e0_a = np.tensordot(e0_a, e0_a, axes=0)
            e0_p = np.tensordot(e0_p, e0_p, axes=0)
            e1_p = np.tensordot(e1_p, e1_p, axes=0)
            e1_a = np.tensordot(e1_a, e1_a, axes=0)

    for i in range(ind + 1, num_of_q):
        e0_a = np.tensordot(e0_a, E, axes=0)
        e0_p = np.tensordot(e0_p, E, axes=0)
        e1_p = np.tensordot(e1_p, E, axes=0)
        e1_a = np.tensordot(e1_a, E, axes=0)

    e0_a = e0_a.reshape((2 ** num_of_q, 2 ** num_of_q))
    e0_p = e0_p.reshape((2 ** num_of_q, 2 ** num_of_q))
    e1_p = e1_p.reshape((2 ** num_of_q, 2 ** num_of_q))
    e1_a = e1_a.reshape((2 ** num_of_q, 2 ** num_of_q))

    for i in range(2 ** num_of_q, dim):
        new_row = np.zeros((1, i + 1), dtype=np.cdouble)
        new_row[0][i] = 1
        e0_a = np.concatenate([e0_a, np.zeros((i, 1), dtype=np.cdouble)], axis=1)
        e0_a = np.concatenate([e0_a, new_row], axis=0)
        e1_a = np.concatenate([e1_a, np.zeros((i, 1), dtype=np.cdouble)], axis=1)
        e1_a = np.concatenate([e1_a, np.zeros((1, i + 1), dtype=np.cdouble)], axis=0)
        e0_p = np.concatenate([e0_p, np.zeros((i, 1), dtype=np.cdouble)], axis=1)
        e0_p = np.concatenate([e0_p, new_row], axis=0)
        e1_p = np.concatenate([e1_p, np.zeros((i, 1), dtype=np.cdouble)], axis=1)
        e1_p = np.concatenate([e1_p, np.zeros((1, i + 1), dtype=np.cdouble)], axis=0)

    # print("a0: ", e0_a)
    # print("a1: ", e1_a)
    # print("p0: ", e0_p)
    # print("p1: ", e1_p)

    e0_a = tf.convert_to_tensor(e0_a, dtype=COMPLEX)
    e0_p = tf.convert_to_tensor(e0_p, dtype=COMPLEX)
    e1_p = tf.convert_to_tensor(e1_p, dtype=COMPLEX)
    e1_a = tf.convert_to_tensor(e1_a, dtype=COMPLEX)

    e0_a_channel = c_util.convert_1qmatrix_to_channel(e0_a)
    e0_p_channel = c_util.convert_1qmatrix_to_channel(e0_p)
    e1_p_channel = c_util.convert_1qmatrix_to_channel(e1_p)
    e1_a_channel = c_util.convert_1qmatrix_to_channel(e1_a)

    tf.print("e0_a: ", tf.math.real(tf.math.reduce_sum(e0_a_channel[0])), output_stream=sys.stdout)
    # tf.print("element sum: ", tf.math.real(tf.math.reduce_sum(e0_a_channel)), output_stream=sys.stdout)

    tf.print("a0: ", tf.math.real(e0_a), output_stream=sys.stdout)
    # tf.print("a1: ", tf.math.real(e1_a), output_stream=sys.stdout)
    # tf.print("p0: ", tf.math.real(e0_p), output_stream=sys.stdout)
    # tf.print("p1: ", tf.math.real(e1_p), output_stream=sys.stdout)

    # TODO: check correctness
    ap_channel = (e0_a_channel + e1_a_channel) @ (e0_p_channel + e1_p_channel)
    tf.print("DebugPrint: ", tf.math.real(ap_channel), output_stream=sys.stdout)'''

    return ap_channel


@tf.function
def make_1q_hybrid_channel(target: TENSOR, args_list: TENSOR, dim: int = 2, ind: int = 0) -> TENSOR:
    """
    Args:
        target: a Tensor(4,4)[complex128] - a channel, which we are noising now will be applied
        args_list: a Tensor(3)[float] containing arguments for applying noise models.
        First arg is p for depolarization, and args 2 & 3 are for gamma1 & gamma2 - params for APD
        ind: index of target qubit inside qudit
        dim: number of dimensions of qudit

    Returns:
        Tensor(4,4)[complex128] - new noised channel
    """
    E = tf.eye(dim, dtype=COMPLEX)
    E_channel = c_util.convert_1qmatrix_to_channel(E)
    p = tf.cast(args_list[0], COMPLEX)
    gamma1 = args_list[1] / 2
    gamma2 = args_list[2] / 2

    ap_channel = create_AP_matrix(gamma1, gamma2, dim, ind)

    output = (target * (1 - p) +
              c_util.convert_1qmatrix_to_channel(create_sigmaX(dim, ind)) * p * 0.25 +
              E_channel * p * 0.25 +
              c_util.convert_1qmatrix_to_channel(create_sigmaY(dim, ind)) * p * 0.25 +
              c_util.convert_1qmatrix_to_channel(create_sigmaZ(dim, ind)) * p * 0.25)
    output = ap_channel @ output @ ap_channel

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
def make_1q_4pars_channel(target: TENSOR, args_list: list[float], dim: int = 2, ind: int = 0) -> TENSOR:
    """
    TODO: Write docstring
    """
    # assert (len(args_list) == 4)
    # p_dep, gamma1, gamma2, sigma = args_list

    disp_channel = create_1q_dispersed_channel(target, args_list[0], dim)
    output = make_1q_hybrid_channel(disp_channel, args_list[1:], dim, ind)

    return output


@tf.function
def make_2q_4pars_channel(target: TENSOR, args_list: list[float]) -> TENSOR:
    """
    TODO: Write docstring
    """
    # assert (len(args_list) == 4)
    # p_dep, gamma1, gamma2, sigma = args_list
    disp_channel = create_2q_dispersed_channel(target, args_list[0])
    output = make_2q_hybrid_channel(disp_channel, args_list[1:])

    return output
