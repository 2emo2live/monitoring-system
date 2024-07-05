import tensorflow as tf  # tf 2.x
import tensornetwork as tn
from tensornetwork import ncon
import typing as tp

from solver.utils.misc import NconTemplate, COMPLEX, FLOAT

tn.set_default_backend("tensorflow")


# TODO: rename samples to outcomes to be consistent with the article

class QCEvaluator:
    def __init__(self, gates: list[tf.Tensor],  n: int, dim: int = 2):
        """
        A class which evaluates quantum circuits with support for d-dimensional qudits. 
        It can generate a circuit's outcomes or evaluate the probability of outcomes.

        Args:
            gates: List of Tensors representing quantum gates. Each gate should be
                compatible with the specified dimension 'dim'.
            
            n: Number of qudits in the circuit.

            dim: Dimensionality of each qudit. Default is 2, which corresponds to qubits.
                For dim > 2, the qudits are d-dimensional.

        Attributes:
            gates: List of quantum gates used in the circuits.
            
            circuits: Dictionary of names and ncon templates specifying each circuit.
                    Unlike the default ncon format, tensors are identified by their
                    positions in the 'gates' attribute, not contained explicitly.
            
            n: Number of qudits in the circuit.

            dim: Dimensionality of each qudit in the circuit.

            in_states: List of quantum states serving as inputs for the quantum circuit.
                    Initially, all qudits are set to the |0⟩ state in their respective
                    d-dimensional space, represented as a vector of length dim^2 with
                    the first element being 1 and the rest being 0.
        """

        self.gates = gates
        self.circuits: dict[str, NconTemplate] = {}
        self.n = n
        self.dim = dim
        self.in_states = n * [tf.constant([1] + (self.dim ** 2 - 1) * [0], dtype=COMPLEX)]

    def add_circuit(self, tn_template: NconTemplate, name: str) -> None:
        """
        Adds a circuit to a class attribute. Keep in mind the format!
        """
        self.circuits[name] = tn_template

    # TODO: try to insert @tf.function here for speed
    def evaluate(self, samples: tf.Tensor, name: str) -> tf.Tensor:
        """
        Evaluates probabilities of obtaining each sample in 'samples' as the circuit's action upon input states.

        Args:
            name: name of the circuit
            samples: Tensor(bs, self.n)[int32] - batch of bitstrings

        Returns:
            Tensor(bs)[complex128] - batch of probabilities, one for each bitstring in 'samples'
        """
        # (bs, n, d^2); n - enumerates a tensor
        out_tensors = tf.one_hot(tf.multiply(samples, self.dim**2 - 1), self.dim**2, dtype=COMPLEX)
        out_tensors = [out_tensors[:, i, :] for i in range(self.n)]

        tensors, net_struc, con_order, out_order = self.circuits[name]
        tensors = out_tensors + self.in_states + [self.gates[i] for i in tensors]

        for i, arr in enumerate(net_struc):
            for j, obj in enumerate(arr):
                if isinstance(obj, int):
                    if obj < 0:
                        net_struc[i][j] = 'out' + str(-net_struc[i][j])  # don't use obj - it's a copy

        net_struc = ([[-1, 'out' + str(i)] for i in range(1, self.n + 1)]
                     + [[i] for i in range(1, self.n + 1)]) + net_struc
        con_order: list[tp.Union[str, int]] = (['out' + str(i) for i in range(1, self.n + 1)]
                                               + list(range(1, self.n + 1))) + con_order

        return ncon(tensors, net_struc, con_order, (-1,))

    def sample_next_qubit(self, name: str, prev_samples: tp.Optional[tf.Tensor], bs_override: int = 1000):
        """
        Creates a batch of samples for a circuit with name 'name' for a single qubit no. L
        Samples are represented by a Tensor(bs,)[int32], but the final goal is to make a Tensor(bs, self.n)[int32]
        Since we employ 'qubit-by-qubit' sample generation, L is defined as (prev_samples.shape[1] + 1)
        if prev_samples is None, this means we start the generation from 1st qubit

        Args:
            name: name of the circuit in the QuantumCircuits class instance
            prev_samples: None or Tensor(bs, (L-1))[int32].
            Contains previously generated bitstrings for first (L-1) qubits
            bs_override: sample size. is needed when starting from scratch (default value is 1000).

        Returns:
            1D array of type int32 of shape (bs)
        """
        # this block defines the qubit l we need to sample, extracts the batch size and prev_samples is they exist
        # we use one_hot(prev_samples) for out (l-1) qubits which are already sampled, and connect them to '-1'
        # we use tf.eye(4) for qubit number l which we sample at the moment and connect it to '-2'

        if prev_samples is not None:
            qubit_id = prev_samples.shape[1] + 1
            bs = prev_samples.shape[0]
            one_hot_prev_samples = tf.one_hot(tf.multiply(prev_samples, self.dim**2 - 1), self.dim**2, dtype=COMPLEX)  # bs, l, 4
            slices = [one_hot_prev_samples[:, i, :] for i in range(0, qubit_id - 1)]  # 1..(l-1)
            out_tensors = slices + [tf.eye(self.dim**2, dtype=COMPLEX)]  # slices & target
            out_new_order = (-1, -2)
        else:
            qubit_id = 1
            bs = bs_override
            # slices = None
            out_tensors = [tf.eye(self.dim**2, dtype=COMPLEX)]  # slices & target
            out_new_order = (-2,)

        # now we take care about plugs - qubits after l which are going to be sampled later
        plug_state = [1] + [0] * (self.dim**2 - 1)
        plugs = (self.n - qubit_id) * [tf.constant(plug_state, dtype=COMPLEX)]
        out_tensors = out_tensors + plugs

        # we unpack a tensor network template, then add slices, add target qubit, add plugs, and finally input legs
        tensors, net_struc, con_order, out_order = self.circuits[name]
        tensors = out_tensors + self.in_states + [self.gates[i] for i in tensors]

        for i, arr in enumerate(net_struc):
            for j, obj in enumerate(arr):
                if isinstance(obj, int):
                    if obj < 0:
                        net_struc[i][j] = 'out' + str(-net_struc[i][j])

        net_struc = ([[-1, 'out' + str(i)] for i in range(1, qubit_id)] +
                     [[-2, 'out' + str(qubit_id)]] +
                     [['out' + str(i)] for i in range(qubit_id + 1, self.n + 1)] +
                     [[i] for i in range(1, self.n + 1)]) + net_struc

        con_order: list[tp.Union[str, int]] = (['out' + str(i) for i in range(1, self.n + 1)]
                                               + list(range(1, self.n + 1))) + con_order

        psi = ncon(tensors, net_struc, con_order, out_new_order)
        psi = tf.abs(psi)

        # Отладочный вывод для размерности psi
        # print("Debug: psi shape:", psi.shape)

        # if psi.shape[0] == 0:
        #     raise ValueError("psi is empty, which indicates an issue with tensor network contraction.")

       
        if prev_samples is not None:
            big_p = tf.concat([psi[:, 0][tf.newaxis], psi[:, self.dim**2 - 1][tf.newaxis]], axis=0)
            big_p = tf.transpose(big_p)
            big_p = big_p / tf.reduce_sum(big_p, axis=1, keepdims=True)
            log_probs = tf.math.log(big_p)  # Gumbel trick (Ilya is a genius)
            eps = -tf.math.log(-tf.math.log(tf.random.uniform(log_probs.shape, dtype=FLOAT)))
            samples = (tf.argmax(log_probs + eps, axis=-1, output_type=tf.int32))
        else:
            big_p = tf.concat([psi[0][tf.newaxis], psi[self.dim**2 - 1][tf.newaxis]], axis=0)
            big_p = big_p / tf.reduce_sum(big_p, keepdims=True)
            log_probs = tf.math.log(big_p)
            eps = -tf.math.log(-tf.math.log(tf.random.uniform((bs, self.dim), dtype=FLOAT))) #TODO: check if 2 or dim
            samples = (tf.argmax(log_probs + eps, axis=-1, output_type=tf.int32))


        return samples


    def make_full_samples(self, name: str, bs_override: int = 1000):
        """
        Creates a Tensor(bs, self.n)[int32] - batch of all-qubit samples for a circuit with name 'name'

        Args:
            name: name of the circuit in the QuantumCircuits class instance
            bs_override: sample size (default value is 1000)

        Returns:
            2D array of type int32 of shape (bs, self.n)
        """
        next_samples = self.sample_next_qubit(name, prev_samples=None, bs_override=bs_override)
        big_samples = next_samples[:, tf.newaxis]

        for _ in range(1, self.n):
            next_samples = self.sample_next_qubit(name, prev_samples=big_samples)
            big_samples = tf.concat([big_samples, next_samples[:, tf.newaxis]], axis=1)
        return big_samples
