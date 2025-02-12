import tensorflow as tf
import scipy
import numpy as np

from solver.utils.misc import COMPLEX


def same_matrix(matrix1: tf.Tensor, matrix2: tf.Tensor, eps: float = 1e-4) -> bool:
    if matrix1.shape != matrix2.shape:
        return False
    norm = tf.abs(tf.linalg.norm(matrix1 - matrix2))
    if norm >= eps:
        print(f"tf.linalg.norm between matrices {norm.numpy()} is too big for eps={eps}!")
        return False

    return True


MIXED_2Q = tf.eye(4, dtype=COMPLEX) / tf.constant(4, dtype=COMPLEX)
MIXED_1Q = tf.eye(2, dtype=COMPLEX) / tf.constant(2, dtype=COMPLEX)


def is_dm(matrix: tf.Tensor, eps: float = 1e-7) -> int:
    """
    Tests whether the input matrix is a vali density matrix
    Returns: 0 if everything is passed
    1 if one is not Hermitian
    2 if one of eigs has imaginary part
    3 if one of eigs is negative
    4 if eigs do not sum to 1
    """

    if not same_matrix(matrix, tf.transpose(tf.math.conj(matrix))):
        return 1

    eigs = scipy.linalg.eigh(matrix)[0]

    for eig in eigs:
        if tf.abs(tf.math.imag(eig)) > 1e-6:  # weakened from 1e-10 to support float32
            print(f"Eig {eig} has too big imaginary part")
            return 2
        if tf.math.real(eig) < -1e-6:
            print(f"Eig {eig} is negative")
            return 3

    if tf.abs(eigs.sum() - 1) >= eps:
        print(f"Sum of eigs {eigs.sum()} is too far from 1: norm = {(tf.abs(eigs.sum() - 1)).numpy()}, eps={eps}")
        return 4

    return 0


def is_choi(matrix: tf.Tensor, eps: float = 1e-5, dim: int = None) -> bool:
    if dim is None:
        dim = int(np.sqrt(matrix.shape[0])) # restoring dimension of a particle from its shape
    assert matrix.shape == (dim**2, dim**2), f'Wrong dimension of input matrix for checking Choiness: {matrix.shape }'

    if is_dm(matrix, eps=eps) != 0:
        return False

    matrix = tf.reshape(matrix, (dim, dim, dim, dim))
    part_trace = tf.einsum('ijik->jk', matrix)

    mixed = tf.eye(dim, dtype=COMPLEX) / tf.constant(dim, dtype=COMPLEX)
    return same_matrix(part_trace, mixed, eps=eps)


def create_random_channel(qubits: int, d: int = 2) -> tf.Tensor:
    """
    ONLY FOR TESTING PURPOSES
    """
    dim = d ** qubits
    dim_squared = dim ** 2
    kraus_rank = np.random.randint(1, dim_squared)
    kraus_ops: list[np.array] = []

    for i in range(kraus_rank):
        rnd_matrix = np.random.rand(dim, dim) * 1j / dim_squared
        rnd_matrix += np.random.rand(dim, dim) / dim_squared
        kraus_ops.append(rnd_matrix)

    sum_parts = np.eye(dim, dtype=np.complex128)
    for op in kraus_ops:
        sum_parts -= np.conj(op.T) @ op
    kraus_ops.append(scipy.linalg.sqrtm(sum_parts))

    sanity_check = np.eye(dim, dtype=np.complex128)
    for op in kraus_ops:
        sanity_check -= np.conj(op.T) @ op
    assert np.abs(scipy.linalg.norm(sanity_check)) < 1e-5

    ans = np.zeros((dim_squared, dim_squared), dtype=np.complex128)
    for op in kraus_ops:
        ans += np.reshape(np.einsum('ij,kl->ikjl', op, op.conj()), (dim_squared, dim_squared))

    return tf.convert_to_tensor(ans, dtype=COMPLEX)
