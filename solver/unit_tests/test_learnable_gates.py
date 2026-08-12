import tensorflow as tf
import numpy as np
import typing as tp
from scipy.stats import unitary_group
import pytest

import solver.utils.general_utils as util
import solver.utils.channel_utils as c_util
import solver.noising_tools as ns
from solver.QCSolver import QGOptSolver, get_complex_channel_form
from solver.utils.misc import COMPLEX, FLOAT, ID_GATE
import solver.circuits_generation as cg
import QGOpt as qgo


def create_test_gates(dim: int = 2) -> dict[str, tf.Tensor]:
    """Create simple test gates for testing."""
    # Identity gate
    identity = tf.eye(dim, dtype=COMPLEX)
    identity_channel = c_util.convert_1qmatrix_to_channel(identity)

    U1 = c_util.convert_1qmatrix_to_channel(unitary_group.rvs(dim))
    U2 = c_util.convert_1qmatrix_to_channel(unitary_group.rvs(dim))
    V = c_util.convert_2qmatrix_to_channel(unitary_group.rvs(dim ** 2).reshape((dim, dim, dim, dim)), dim=dim)
    gate_dict = {
        "U1": U1,
        "U2": U2,
        ID_GATE: identity_channel,
        "V": V,
    }

    return gate_dict


def test_learnable_gate_types_initialization():
    """Test that QGOptSolver correctly initializes learnable gate types."""
    for dim in [2, 3, 4]:
        n_qudits = 3
        pure_channels = create_test_gates(dim)

        # Test 1: All gates learnable by default (when learnable_gates_names is None)
        solver1 = QGOptSolver(
            qudits_num=n_qudits,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names=None,  # Default: all learnable
            dim=dim
        )

        # Check that all gates are Variables (learnable)
        for gate_name, gate_tensor in solver1.estimated_gates_dict.items():
            assert isinstance(gate_tensor, tf.Variable), f"{gate_name} should be Variable when all gates learnable"

        # Test 2: Only specific gate types learnable
        learnable_gates = {'U2', 'V'}  # Only RX and CX gates are learnable, H and ID are fixed
        solver2 = QGOptSolver(
            qudits_num=n_qudits,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names=learnable_gates,
            dim=dim
        )

        # Check RX and CX are Variables (learnable)
        assert isinstance(solver2.estimated_gates_dict['U2'], tf.Variable), "RX should be Variable"
        assert isinstance(solver2.estimated_gates_dict['V'], tf.Variable), "CX should be Variable"

        # Check H and ID are not Variables (fixed)
        assert not isinstance(solver2.estimated_gates_dict['U1'], tf.Variable), "H should be constant"

        # Test 3: No gates learnable (empty set)
        solver3 = QGOptSolver(
            qudits_num=n_qudits,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names=set(),  # Empty set: no gates learnable
            dim=dim
        )

        # No gates should be Variables
        for gate_name, gate_tensor in solver3.estimated_gates_dict.items():
            if gate_name == ID_GATE:
                continue
            assert not isinstance(gate_tensor, tf.Variable), f"{gate_name} should be constant when no gates learnable"


def test_learnable_gate_types_gradient_computation():
    """Test that gradients are only computed for learnable gate types."""
    for dim in [2, 3, 4]:
        n_qudits = 2
        pure_channels = create_test_gates(dim)

        # Create circuits for testing
        gen = cg.DataGenerator(
            qubits_num=n_qudits,
            gates_names=['U1', 'U2', 'V'],
            single_qub_gates_num=2,
            two_qub_gates_num=1
        )

        circuits = [['U1_0', 'U2_1', 'V_0_1']]
        ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)

        # Test with mixed learnable gates: only RX learnable
        learnable_gates = {'U2'}  # Only RX gates are learnable

        solver = QGOptSolver(
            qudits_num=n_qudits,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names=learnable_gates,
            dim=dim,
            noise_iter0=0.1
        )

        # Add circuit
        for name, tmpl in ncon_tmpls.items():
            solver.add_circuit(tn_template=tmpl, name=name)

        # Generate samples
        solver.generate_all_samples(smpl_size=1000, v=False)

        # Compute loss and gradients
        loss, grad_dict = solver._loss_and_grad(lmbd1=1.0, lmbd2=1.0)

        # Check that loss is a tensor
        assert isinstance(loss, tf.Tensor)

        assert 'U2' in grad_dict, "U2 should have gradients"
        assert grad_dict['U2'] is not None, "U2 gradient should not be None"

        assert 'U1' not in grad_dict, "U1 shouldn't have gradients"
        assert 'V' not in grad_dict, "V shouldn't have gradients"


def test_learnable_gate_types_optimization():
    """Test that only learnable gate types are updated during optimization."""
    for dim in [2, 3, 4]:
        n_qudits = 2
        pure_channels = create_test_gates(dim)

        # Create simple circuit
        gen = cg.DataGenerator(
            qubits_num=n_qudits,
            gates_names=['U1', 'U2', 'V'],
            single_qub_gates_num=2,
            two_qub_gates_num=1
        )

        circuits = [['U1_0', 'U2_1', 'V_0_1']]
        ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)

        # Only RX gates are learnable
        learnable_gates = {'U2'}

        solver = QGOptSolver(
            qudits_num=n_qudits,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names=learnable_gates,
            dim=dim,
            noise_iter0=0.05
        )

        # Add circuit
        for name, tmpl in ncon_tmpls.items():
            solver.add_circuit(tn_template=tmpl, name=name)

        # Generate samples
        solver.generate_all_samples(smpl_size=500, v=False)

        # Store initial values
        initial_values = {}
        for gate_name, gate_tensor in solver.estimated_gates_dict.items():
            initial_values[gate_name] = gate_tensor.numpy().copy()

        # Create optimizer and run a few iterations
        manif = qgo.manifolds.ChoiMatrix()
        opt = qgo.optimizers.RAdam(manif, 0.01)

        loss_dynamics = solver.train_optimizer(
            opt=opt,
            lmbd1=1.0,
            lmbd2=1.0,
            iters=5,
            v=0
        )

        # Check that loss changed
        assert len(loss_dynamics) == 5
        # Loss might not always decrease due to regularization, but should change
        assert not np.allclose(loss_dynamics, loss_dynamics[0], atol=1e-10)

        # Check which gates were updated
        for gate_name, gate_tensor in solver.estimated_gates_dict.items():
            current_value = gate_tensor.numpy()
            initial_value = initial_values[gate_name]

            if gate_name == 'U2':  # Learnable gate type
                # Should have changed (though change might be small)
                assert not np.allclose(current_value, initial_value, atol=1e-6), \
                    f"Learnable gate type {gate_name} should have changed"
            else:
                # Should not have changed (or changed very little due to numerical issues)
                assert np.allclose(current_value, initial_value, atol=1e-6), \
                    f"Fixed gate type {gate_name} should not have changed significantly"


def test_learnable_gate_types_with_circuits():
    """Test learnable gate types with actual circuit evaluation."""
    for dim in [2, 3, 4]:
        n_qudits = 2
        pure_channels = create_test_gates(dim)

        # Create multiple circuits
        gen = cg.DataGenerator(
            qubits_num=n_qudits,
            gates_names=['U1', 'U2', 'V'],
            single_qub_gates_num=2,
            two_qub_gates_num=1
        )

        circuits = [
            ['U1_0', 'U2_1'],
            ['U2_0', 'V_0_1'],
            ['U1_0', 'U2_1', 'V_0_1']
        ]
        ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)

        # Only U1 gates are learnable
        learnable_gates = {'U1'}

        solver = QGOptSolver(
            qudits_num=n_qudits,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names=learnable_gates,
            dim=dim,
            noise_iter0=0.02,
            compress_samples=True
        )

        # Add all circuits
        for name, tmpl in ncon_tmpls.items():
            solver.add_circuit(tn_template=tmpl, name=name)

        # Generate samples for all circuits
        solver.generate_all_samples(smpl_size=300, v=False)

        # Test circuit evaluation works with mixed gate types
        for name in solver.tn_templates:
            # Get L1 norms (this internally evaluates circuits)
            l1_pure, l1_true = solver.get_circ_l1_norms(name)
            assert isinstance(l1_pure, tf.Tensor)
            assert isinstance(l1_true, tf.Tensor)
            assert l1_pure.shape == ()
            assert l1_true.shape == ()

        # Test statistics computation
        names_list = list(solver.tn_templates.keys())[:2]  # First 2 circuits
        stats = solver.get_statistic_for_circ(
            names=names_list,
            statistic='l1',
            replace_gates=False,
            v=False
        )

        assert len(stats) == len(names_list)
        for stat in stats:
            assert isinstance(stat, tf.Tensor)


def test_learnable_gate_types_integration():
    """Integration test for learnable gate types feature."""
    for dim in [2, 3, 4]:
        n_qudits = 2
        pure_channels = create_test_gates(dim)

        # Create a realistic scenario
        gen = cg.DataGenerator(
            qubits_num=n_qudits,
            gates_names=['U1', 'U2', 'V'],
            single_qub_gates_num=2,
            two_qub_gates_num=1
        )

        # Generate multiple circuits
        ncon_tmpls = gen.generate_data(
            circ_len_min=3,
            circ_len_max=6,
            circ_num=5,
            two_qubit_gate_prob=0.3,
            custom_name='circ_'
        )

        learnable_gates = {'U1', 'V'}

        noise_cfg = []
        noise_cfg.append(('U1', 0, ns.make_1q_4pars_channel, 0.02, 0.02, 0.02, dim))
        noise_cfg.append(('U1', 1, ns.make_1q_4pars_channel, 0.04, 0.01, 0.03, dim))
        noise_cfg.append(('V', 0, ns.make_2q_4pars_channel, 0.05, 0.02, 0.02, dim))

        solver = QGOptSolver(
            qudits_num=n_qudits,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names=learnable_gates,
            dim=dim,
            noise_iter0=0.03
        )

        # Add all circuits
        for name, tmpl in ncon_tmpls.items():
            solver.add_circuit(tn_template=tmpl, name=name)

        # Generate samples
        solver.generate_all_samples(smpl_size=400, v=False)

        # Test the full training loop
        manif = qgo.manifolds.ChoiMatrix()
        opt = qgo.optimizers.RAdam(manif, 0.005)

        # Train for a few iterations
        iterations = 3
        loss_dynamics = solver.train_optimizer(
            opt=opt,
            lmbd1=0.5,
            lmbd2=1.0,
            iters=iterations,
            v=0
        )

        # Check results
        assert len(loss_dynamics) == iterations

        # Test that we can still evaluate circuits after training
        for name in solver.tn_templates:
            l1_pure, l1_true = solver.get_circ_l1_norms(name)
            assert l1_pure.numpy() >= 0, "L1 norm should be non-negative"
            assert l1_true.numpy() >= 0, "L1 norm should be non-negative"


def test_tied_gate_single_shared_instance():
    """Tied gates own one learnable instance, broadcast to all qudit pairs at eval."""
    for dim in [2, 3]:
        n_qudits = 2
        pure_channels = create_test_gates(dim)
        gen = cg.DataGenerator(
            qubits_num=n_qudits,
            gates_names=['U1', 'U2', 'V'],
            single_qub_gates_num=2,
            two_qub_gates_num=1,
        )

        circuits = [['U1_0', 'U2_1', 'V_0_1']]
        ncon_tmpls = gen.get_tmpl_dict_from_human_circs(circuits)

        solver = QGOptSolver(
            qudits_num=n_qudits,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names={'V'},
            tied_gates_names={'V'},
            dim=dim,
            noise_iter0=0.1,
        )

        for name, tmpl in ncon_tmpls.items():
            solver.add_circuit(tn_template=tmpl, name=name)

        # The tied gate owns exactly one learned instance
        assert solver.estimated_gates_dict['V'].shape[0] == 1, "tied gate must have a single instance"

        # Non-tied gates keep per-qudit instances
        assert solver.estimated_gates_dict['U1'].shape[0] == n_qudits

        # At evaluation time the tied channel is expanded to one copy per qudit pair
        channels = solver._get_estimated_channels()
        assert channels['V'].shape[0] == n_qudits * (n_qudits - 1)
        for i in range(1, channels['V'].shape[0]):
            np.testing.assert_allclose(channels['V'][i].numpy(), channels['V'][0].numpy(),
                                       rtol=1e-6, atol=1e-6)

        solver.generate_all_samples(smpl_size=400, v=False)
        loss, grad_dict = solver._loss_and_grad(lmbd1=1.0, lmbd2=1.0)

        # The tied variable receives gradients from all its slots
        assert grad_dict['V'] is not None, "tied gate should receive gradients"
        assert np.abs(grad_dict['V'].numpy()).sum() > 0, "tied gate gradient should be non-zero"


def test_tied_gate_must_be_learnable():
    """Tied gates must belong to the learnable set."""
    dim = 2
    pure_channels = create_test_gates(dim)
    with pytest.raises(ValueError, match="Tied gates must be learnable"):
        QGOptSolver(
            qudits_num=2,
            single_qud_gates_names={'U1', 'U2'},
            two_qud_gates_names={'V'},
            pure_channels_set=pure_channels,
            learnable_gates_names={'U1'},
            tied_gates_names={'V'},
            dim=dim,
        )
