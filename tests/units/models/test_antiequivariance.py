"""Test model antiequivariances."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import vmcnet.models as models
import vmcnet.models.antiequivariance as antieq
from vmcnet.utils.typing import SLArray, SpinSplitSLArray
from vmcnet.utils.slog_helpers import (
    array_to_slog,
    array_from_slog,
    slog_sum,
)

from .utils import get_elec_hyperparams, get_input_streams_from_hyperparams
from tests.test_utils import assert_pytree_allclose


def test_slog_cofactor_output_with_batches():
    """Test slog_cofactor_antieq outputs correct value on simple inputs."""
    input = jnp.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    negative_input = -input
    doubled_input = input * 2
    full_input = jnp.stack([input, negative_input, doubled_input])

    expected_out = jnp.array([-3, 12, -9])
    expected_signs = jnp.sign(expected_out)
    expected_logs = jnp.log(jnp.abs(expected_out))
    full_expected_signs = jnp.stack([expected_signs, -expected_signs, expected_signs])
    full_expected_logs = jnp.stack(
        [expected_logs, expected_logs, expected_logs + jnp.log(8)]
    )

    y = antieq.slog_cofactor_antieq(full_input)

    np.testing.assert_allclose(y[0], full_expected_signs)
    np.testing.assert_allclose(y[1], full_expected_logs, rtol=1e-6)


def test_slog_cofactor_antiequivarance():
    """Test slog_cofactor_antieq is antiequivariant."""
    input = jnp.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    permutation = jnp.array([1, 0, 2])
    perm_input = input[permutation, :]

    output_signs, output_logs = antieq.slog_cofactor_antieq(input)
    perm_signs, perm_logs = antieq.slog_cofactor_antieq(perm_input)

    expected_perm_signs = -output_signs[permutation]
    expected_perm_logs = output_logs[permutation]

    np.testing.assert_allclose(perm_signs, expected_perm_signs)
    np.testing.assert_allclose(perm_logs, expected_perm_logs)


def test_orbital_cofactor_layer_antiequivariance():
    """Test evaluation and antiequivariance of orbital cofactor equivariance layer."""
    # Generate example hyperparams and input streams
    (
        nchains,
        nelec_total,
        nion,
        d,
        permutation,
        spin_split,
        split_perm,
    ) = get_elec_hyperparams()
    (
        input_1e,
        _,
        input_ei,
        perm_input_1e,
        _,
        perm_input_ei,
        key,
    ) = get_input_streams_from_hyperparams(nchains, nelec_total, nion, d, permutation)

    # Set up antiequivariant layer
    kernel_initializer = models.weights.get_kernel_initializer("orthogonal")
    bias_initializer = models.weights.get_bias_initializer("normal")
    orbital_cofactor_antieq = antieq.OrbitalCofactorAntiequivarianceLayer(
        spin_split,
        kernel_initializer,
        kernel_initializer,
        kernel_initializer,
        bias_initializer,
    )
    params = orbital_cofactor_antieq.init(key, input_1e, input_ei)

    # Evaluate layer on original and permuted inputs
    output = orbital_cofactor_antieq.apply(params, input_1e, input_ei)
    perm_output = orbital_cofactor_antieq.apply(params, perm_input_1e, perm_input_ei)

    # Verify output shape and verify all signs values are  +-1
    nelec_per_spin = models.core.get_nelec_per_spin(spin_split, nelec_total)
    nspins = len(nelec_per_spin)
    assert len(output) == nspins
    for i in range(nspins):
        assert len(output[i]) == 2
        d_input_1e = input_1e.shape[-1]
        np.testing.assert_allclose(
            jnp.abs(output[i][0]),
            jnp.ones((nchains, nelec_per_spin[i], d_input_1e)),
        )
        chex.assert_shape(output[i][1], (nchains, nelec_per_spin[i], d_input_1e))

    # Verify that permutation has generated appropriate antiequivariant transformation
    flips = [-1, 1]  # First spin permutation is odd; second is even
    for i in range(nspins):
        signs, logs = output[i]
        perm_signs, perm_logs = perm_output[i]
        expected_perm_signs = signs[:, split_perm[i], :] * flips[i]
        expected_perm_logs = logs[:, split_perm[i], :]
        np.testing.assert_allclose(perm_signs, expected_perm_signs)
        np.testing.assert_allclose(perm_logs, expected_perm_logs)


def test_get_odd_symmetries_one_spin():
    """Test odd symmetry generation for a single spin."""
    nspins = 3
    signs = jnp.array([[[1, 1], [1, -1]], [[-1, 1], [1, 1]]])
    logs = jnp.array([[[-2.4, 2.4], [1.6, 0.8]], [[0.3, -0.2], [-10, 0]]])

    # Test for ispin=0, sign should alternate at each index
    (sym_signs, sym_logs) = antieq.get_odd_symmetries_one_spin((signs, logs), 0, nspins)
    expected_signs = jnp.stack(
        [signs, -signs, signs, -signs, signs, -signs, signs, -signs],
        axis=-3,
    )
    expected_logs = jnp.stack([logs, logs, logs, logs, logs, logs, logs, logs], axis=-3)
    np.testing.assert_allclose(sym_signs, expected_signs)
    np.testing.assert_allclose(sym_logs, expected_logs)

    # Test for ispin=1, sign should alternate every two indices
    (sym_signs, sym_logs) = antieq.get_odd_symmetries_one_spin((signs, logs), 1, nspins)
    expected_signs = jnp.stack(
        [signs, signs, -signs, -signs, signs, signs, -signs, -signs],
        axis=-3,
    )
    np.testing.assert_allclose(sym_signs, expected_signs)
    np.testing.assert_allclose(sym_logs, expected_logs)

    # Test for ispin=3, sign should alternate every four indices
    (sym_signs, sym_logs) = antieq.get_odd_symmetries_one_spin((signs, logs), 2, nspins)
    expected_signs = jnp.stack(
        [signs, signs, signs, signs, -signs, -signs, -signs, -signs],
        axis=-3,
    )
    np.testing.assert_allclose(sym_signs, expected_signs)
    np.testing.assert_allclose(sym_logs, expected_logs)


def test_get_all_odd_symmetries():
    """Test odd symmetry generation for multiple spins."""
    spin1_signs = jnp.array([[[1, 1], [1, -1]], [[-1, 1], [1, 1]]])
    spin2_signs = jnp.array([[[1, -1], [-1, -1], [1, 1]], [[-1, 1], [-1, 1], [1, -1]]])
    spin1_logs = jnp.array([[[-2.4, 2.4], [1.6, 0.8]], [[0.3, -0.2], [-10, 0]]])
    spin2_logs = jnp.array(
        [[[-2.9, 2.0], [1.2, 0.8], [0.3, 0.2]], [[0.3, -0.2], [-10, 0], [-11, -1]]]
    )
    inputs = [(spin1_signs, spin1_logs), (spin2_signs, spin2_logs)]

    # Test for ispin=0, sign should alternate at each index
    syms = antieq.get_all_odd_symmetries(inputs)
    expected_syms = [
        (
            jnp.stack([spin1_signs, -spin1_signs, spin1_signs, -spin1_signs], axis=-3),
            jnp.stack([spin1_logs, spin1_logs, spin1_logs, spin1_logs], axis=-3),
        ),
        (
            jnp.stack([spin2_signs, spin2_signs, -spin2_signs, -spin2_signs], axis=-3),
            jnp.stack([spin2_logs, spin2_logs, spin2_logs, spin2_logs], axis=-3),
        ),
    ]
    assert_pytree_allclose(syms, expected_syms)


def test_sum_sl_array():
    """Test sum_sl_array helper function."""
    vals = jnp.array([[-1, -2, -3, 6], [1, 2, 3, 4], [0.3, 0.4, 6, -2]])
    expected_sums = jnp.array([0, 10, 4.7])
    sl_vals = array_to_slog(vals)
    expected_sl_sums = array_to_slog(expected_sums)

    sl_sums = slog_sum(sl_vals, axis=-1)
    np.testing.assert_allclose(sl_sums, expected_sl_sums)


def test_make_fn_odd_one_spin():
    """Test making a function of one spin odd."""
    nbatch = 5
    nelec = 3
    d = 2
    dhidden = 4
    dout = 3
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    inputs = jax.random.normal(key, (nbatch, nelec, d))
    neg_inputs = -inputs
    weights1 = jax.random.normal(subkey, (nelec * d, dhidden))
    weights2 = jax.random.normal(subkey, (dhidden, dout))
    slog_inputs = [array_to_slog(inputs)]  # one spin block only
    neg_slog_inputs = [array_to_slog(neg_inputs)]

    def fn(x: SpinSplitSLArray) -> SLArray:
        (signs, logs) = x[0]

        values = signs * jnp.exp(logs)
        values = values.reshape((values.shape[0], values.shape[1], nelec * d))
        output = jnp.matmul(values, weights1)
        output = jnp.tanh(output)
        output = jnp.matmul(output, weights2)
        output = jnp.tanh(output)
        return array_to_slog(output)

    odd_fn = antieq.make_fn_odd(fn)
    result = odd_fn(slog_inputs)
    neg_result = odd_fn(neg_slog_inputs)

    np.testing.assert_allclose(neg_result[0], -result[0])
    np.testing.assert_allclose(neg_result[1], result[1])


def test_make_fn_odd_multi_spin():
    """Test making a function of multiple spins odd with respect to each."""
    nbatch = 5
    nelec_per_spin = (2, 3, 4)
    nelec_total = 9
    d = 2
    dhidden = 4
    dout = 3
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    inputs = [jax.random.normal(key, (nbatch, n, d)) for n in nelec_per_spin]
    flip_sign_inputs = [inputs[0], -inputs[1], inputs[2]]
    same_sign_inputs = [inputs[0], -inputs[1], -inputs[2]]

    weights1 = jax.random.normal(subkey, (nelec_total * d, dhidden))
    weights2 = jax.random.normal(subkey, (dhidden, dout))
    slog_inputs = [antieq.get_sl_values(x) for x in inputs]  # one spin block only
    flip_slog_inputs = [antieq.get_sl_values(x) for x in flip_sign_inputs]
    same_slog_inputs = [antieq.get_sl_values(x) for x in same_sign_inputs]

    def fn(x: SpinSplitSLArray) -> SLArray:
        all_signs = jnp.concatenate([s[0] for s in x], axis=-2)
        all_logs = jnp.concatenate([s[1] for s in x], axis=-2)
        all_vals = array_from_slog((all_signs, all_logs))
        all_vals = jnp.reshape(
            all_vals, (all_vals.shape[0], all_vals.shape[1], nelec_total * d)
        )
        output = jnp.matmul(all_vals, weights1)
        output = jnp.tanh(output)
        output = jnp.matmul(output, weights2)
        output = jnp.tanh(output)
        return array_to_slog(output)

    odd_fn = antieq.make_fn_odd(fn)
    result = odd_fn(slog_inputs)
    flip_sign_result = odd_fn(flip_slog_inputs)
    same_sign_result = odd_fn(same_slog_inputs)

    expected_neg_result = (-result[0], result[1])
    assert_pytree_allclose(flip_sign_result, expected_neg_result, rtol=1e-5)
    assert_pytree_allclose(result, same_sign_result, rtol=1e-5)
