"""Test sign covariance routines."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.models.sign_covariance as sign_cov
from vmcnet.utils.slog_helpers import (
    array_to_slog,
    array_from_slog,
    array_list_to_slog,
)
from vmcnet.utils.typing import SLArray, SLArrayList

from tests.test_utils import assert_pytree_allclose


def test_get_sign_orbit_one_sl_array():
    """Test sign orbit generation for a single array index."""
    n_total = 3
    signs = jnp.array([[[1, 1], [1, -1]], [[-1, 1], [1, 1]]])
    logs = jnp.array([[[-2.4, 2.4], [1.6, 0.8]], [[0.3, -0.2], [-10, 0]]])

    # Test for i=0, sign should alternate at each index
    syms, sym_signs = sign_cov.get_sign_orbit_one_sl_array(
        (signs, logs), 0, n_total, axis=-3
    )
    expected_logs = jnp.stack([logs, logs, logs, logs, logs, logs, logs, logs], axis=-3)
    expected_syms = (
        jnp.stack(
            [signs, -signs, signs, -signs, signs, -signs, signs, -signs],
            axis=-3,
        ),
        expected_logs,
    )
    expected_sym_signs = jnp.array([1, -1, 1, -1, 1, -1, 1, -1])
    assert_pytree_allclose(syms, expected_syms)
    np.testing.assert_allclose(sym_signs, expected_sym_signs)

    # Test for i=1, sign should alternate every two indices
    syms, sym_signs = sign_cov.get_sign_orbit_one_sl_array(
        (signs, logs), 1, n_total, axis=-3
    )
    expected_syms = (
        jnp.stack(
            [signs, signs, -signs, -signs, signs, signs, -signs, -signs],
            axis=-3,
        ),
        expected_logs,
    )
    expected_sym_signs = jnp.array([1, 1, -1, -1, 1, 1, -1, -1])
    assert_pytree_allclose(syms, expected_syms)
    np.testing.assert_allclose(sym_signs, expected_sym_signs)

    # Test for i=3, sign should alternate every four indices
    syms, sym_signs = sign_cov.get_sign_orbit_one_sl_array(
        (signs, logs), 2, n_total, axis=-3
    )
    expected_syms = (
        jnp.stack(
            [signs, signs, signs, signs, -signs, -signs, -signs, -signs],
            axis=-3,
        ),
        expected_logs,
    )
    expected_sym_signs = jnp.array([1, 1, 1, 1, -1, -1, -1, -1])
    assert_pytree_allclose(syms, expected_syms)
    np.testing.assert_allclose(sym_signs, expected_sym_signs)


def test_get_sign_orbit_slog_array_list():
    """Test sign orbit generation for a full SLArrayList."""
    spin1_signs = jnp.array([[[1, 1], [1, -1]], [[-1, 1], [1, 1]]])
    spin2_signs = jnp.array([[[1, -1], [-1, -1], [1, 1]], [[-1, 1], [-1, 1], [1, -1]]])
    spin1_logs = jnp.array([[[-2.4, 2.4], [1.6, 0.8]], [[0.3, -0.2], [-10, 0]]])
    spin2_logs = jnp.array(
        [[[-2.9, 2.0], [1.2, 0.8], [0.3, 0.2]], [[0.3, -0.2], [-10, 0], [-11, -1]]]
    )
    inputs = [(spin1_signs, spin1_logs), (spin2_signs, spin2_logs)]

    syms, sym_signs = sign_cov.get_sign_orbit_slog_array_list(inputs, axis=-3)

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
    expected_sym_signs = jnp.array([1, -1, -1, 1])

    assert_pytree_allclose(syms, expected_syms)
    np.testing.assert_allclose(sym_signs, expected_sym_signs)


def test_make_slog_fn_sign_covariant():
    """Test making a fn of several spins sign covariant with respect to each spin."""
    nbatch = 5
    nelec_per_spin = (2, 3, 4)
    nelec_total = 9
    d = 2
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    inputs = [jax.random.normal(key, (nbatch, n, d)) for n in nelec_per_spin]
    flip_sign_inputs = [inputs[0], -inputs[1], inputs[2]]
    same_sign_inputs = [inputs[0], -inputs[1], -inputs[2]]
    slog_inputs = array_list_to_slog(inputs)
    flip_slog_inputs = array_list_to_slog(flip_sign_inputs)
    same_slog_inputs = array_list_to_slog(same_sign_inputs)

    dhidden = 4
    dout = 3
    weights1 = jax.random.normal(subkey, (nelec_total * d, dhidden))
    weights2 = jax.random.normal(subkey, (dhidden, dout))

    def fn(x: SLArrayList) -> SLArray:
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

    covariant_fn = sign_cov.make_slog_fn_sign_covariant(fn)
    result = covariant_fn(slog_inputs)
    flip_sign_result = covariant_fn(flip_slog_inputs)
    same_sign_result = covariant_fn(same_slog_inputs)

    expected_neg_result = (-result[0], result[1])
    assert_pytree_allclose(flip_sign_result, expected_neg_result, rtol=1e-5)
    assert_pytree_allclose(same_sign_result, result, rtol=1e-5)
