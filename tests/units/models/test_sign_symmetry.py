"""Test sign covariance routines."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing import Callable

import vmcnet.models.sign_symmetry as sign_sym
import vmcnet.models.weights as weights
from vmcnet.utils.slog_helpers import (
    array_from_slog,
    array_to_slog,
    array_list_to_slog,
)
from vmcnet.utils.typing import Array, ArrayList, PRNGKey, SLArray, SLArrayList

from tests.test_utils import assert_pytree_allclose


def test_get_sign_orbit_array_list():
    """Test sign orbit generation for an ArrayList."""
    s1 = jnp.array([[[-2.4, 2.4], [1.6, 0.8]], [[0.3, -0.2], [-10, 0]]])
    s2 = jnp.array(
        [[[-2.9, 2.0], [1.2, 0.8], [0.3, 0.2]], [[0.3, -0.2], [-10, 0], [-11, -1]]]
    )
    s3 = jnp.array([[[-0.3, 0.6]], [[-2, -10]]])
    inputs = [s1, s2, s3]

    syms, sym_signs = sign_sym._get_sign_orbit_array_list(inputs, axis=-2)

    expected_syms = [
        jnp.stack([s1, -s1, s1, -s1, s1, -s1, s1, -s1], axis=-2),
        jnp.stack([s2, s2, -s2, -s2, s2, s2, -s2, -s2], axis=-2),
        jnp.stack([s3, s3, s3, s3, -s3, -s3, -s3, -s3], axis=-2),
    ]
    expected_sym_signs = jnp.array([1, -1, -1, 1, -1, 1, 1, -1])

    assert_pytree_allclose(syms, expected_syms)
    np.testing.assert_allclose(sym_signs, expected_sym_signs)


def test_get_sign_orbit_slog_array_list():
    """Test sign orbit generation for an SLArrayList."""
    spin1_signs = jnp.array([[[1, 1], [1, -1]], [[-1, 1], [1, 1]]])
    spin2_signs = jnp.array([[[1, -1], [-1, -1], [1, 1]], [[-1, 1], [-1, 1], [1, -1]]])
    spin1_logs = jnp.array([[[-2.4, 2.4], [1.6, 0.8]], [[0.3, -0.2], [-10, 0]]])
    spin2_logs = jnp.array(
        [[[-2.9, 2.0], [1.2, 0.8], [0.3, 0.2]], [[0.3, -0.2], [-10, 0], [-11, -1]]]
    )
    inputs = [(spin1_signs, spin1_logs), (spin2_signs, spin2_logs)]

    syms, sym_signs = sign_sym._get_sign_orbit_sl_array_list(inputs, axis=-2)

    expected_syms = [
        (
            jnp.stack([spin1_signs, -spin1_signs, spin1_signs, -spin1_signs], axis=-2),
            jnp.stack([spin1_logs, spin1_logs, spin1_logs, spin1_logs], axis=-2),
        ),
        (
            jnp.stack([spin2_signs, spin2_signs, -spin2_signs, -spin2_signs], axis=-2),
            jnp.stack([spin2_logs, spin2_logs, spin2_logs, spin2_logs], axis=-2),
        ),
    ]
    expected_sym_signs = jnp.array([1, -1, -1, 1])

    assert_pytree_allclose(syms, expected_syms)
    np.testing.assert_allclose(sym_signs, expected_sym_signs)


def _make_simple_nn_layers(
    dinput: int, dout: int, key: PRNGKey
) -> Callable[[Array], Array]:
    dhidden = 4
    key, subkey = jax.random.split(key)
    # Biases important here to avoid starting with a perfectly odd function
    biases = jax.random.normal(subkey, (dinput,))
    key, subkey = jax.random.split(key)
    weights1 = jax.random.normal(key, (dinput, dhidden))
    weights2 = jax.random.normal(subkey, (dhidden, dout))

    def apply_layers(x: Array) -> Array:
        output = jnp.matmul(x + biases, weights1)
        output = jnp.tanh(output)
        output = jnp.matmul(output, weights2)
        output = jnp.tanh(output)
        return output

    return apply_layers


def _test_sign_covariance_or_invariance(is_invariance: bool, atol: float):
    nbatch = 5
    nelec_per_spin = (2, 3, 4)
    nelec_total = 9
    d = 2
    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)
    inputs = [jax.random.normal(key, (nbatch, n * d)) for n in nelec_per_spin]

    sign_change_inputs = [inputs[0], -inputs[1], inputs[2]]
    double_sign_change_inputs = [inputs[0], -inputs[1], -inputs[2]]

    dout = 3
    nn_layers = _make_simple_nn_layers(nelec_total * d, dout, subkey)

    def fn(x: ArrayList) -> Array:
        all_vals = jnp.concatenate(x, axis=-1)
        return nn_layers(all_vals)

    if is_invariance:
        sym_fn = sign_sym.make_array_list_fn_sign_invariant(fn)
    else:
        sym_fn = sign_sym.make_array_list_fn_sign_covariant(fn)

    result = sym_fn(inputs)
    chex.assert_shape(result, (nbatch, dout))
    sign_change_result = sym_fn(sign_change_inputs)
    double_sign_change_result = sym_fn(double_sign_change_inputs)

    assert_pytree_allclose(double_sign_change_result, result, atol=atol)
    if is_invariance:
        assert_pytree_allclose(sign_change_result, result, atol=atol)
    else:
        assert_pytree_allclose(sign_change_result, -result, atol=atol)


@pytest.mark.slow
def test_make_array_list_fn_sign_covariant():
    """Test making a fn of an ArrayList sign-covariant w.r.t each array."""
    _test_sign_covariance_or_invariance(is_invariance=False, atol=1e-6)


@pytest.mark.slow
def test_make_array_list_fn_sign_invariant():
    """Test making a fn of an ArrayList sign-invariant w.r.t each array."""
    _test_sign_covariance_or_invariance(is_invariance=True, atol=1e-6)


@pytest.mark.slow
def test_make_sl_array_list_fn_sign_covariant():
    """Test making a fn of an SLArrayList sign-covariant w.r.t each SLArray."""
    nbatch = 5
    nelec_per_spin = (2, 3, 4)
    nelec_total = 9
    d = 2
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    inputs = [jax.random.normal(key, (nbatch, n * d)) for n in nelec_per_spin]
    flip_sign_inputs = [inputs[0], -inputs[1], inputs[2]]
    same_sign_inputs = [inputs[0], -inputs[1], -inputs[2]]
    slog_inputs = array_list_to_slog(inputs)
    flip_slog_inputs = array_list_to_slog(flip_sign_inputs)
    same_slog_inputs = array_list_to_slog(same_sign_inputs)

    dout = 3
    nn_layers = _make_simple_nn_layers(2 * nelec_total * d, dout, subkey)

    def fn(x: SLArrayList) -> SLArray:
        all_signs = jnp.concatenate([s[0] for s in x], axis=-1)
        all_logs = jnp.concatenate([s[1] for s in x], axis=-1)
        all_vals = jnp.concatenate([all_signs, all_logs], axis=-1)
        all_vals = jnp.reshape(
            all_vals, (all_vals.shape[0], all_vals.shape[1], 2 * nelec_total * d)
        )
        output = nn_layers(all_vals)
        return array_to_slog(output)

    covariant_fn = sign_sym.make_sl_array_list_fn_sign_covariant(fn)
    # Compare final results in the non-log domain for better numerical accuracy and
    # lower permissible tolerances.
    result = array_from_slog(covariant_fn(slog_inputs))
    chex.assert_shape(result, (nbatch, dout))
    flip_sign_result = array_from_slog(covariant_fn(flip_slog_inputs))
    same_sign_result = array_from_slog(covariant_fn(same_slog_inputs))

    np.testing.assert_allclose(flip_sign_result, -result, atol=1e-6)
    np.testing.assert_allclose(same_sign_result, result, atol=1e-6)


def _test_products_sign_covariance(dout: int, use_weights: bool):
    nbatch = 5
    nelec_per_spin = (2, 5)
    d = 2
    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)
    inputs = [jax.random.normal(key, (nbatch, n, d)) for n in nelec_per_spin]

    flip_sign_inputs = [inputs[0], -inputs[1]]
    same_sign_inputs = [-inputs[0], -inputs[1]]

    model = sign_sym.ProductsSignCovariance(
        dout, weights.get_kernel_initializer("orthogonal"), use_weights=use_weights
    )
    result, params = model.init_with_output(subkey, inputs)

    chex.assert_shape(result, (nbatch, dout))
    flip_sign_result = model.apply(params, flip_sign_inputs)
    same_sign_result = model.apply(params, same_sign_inputs)

    assert_pytree_allclose(flip_sign_result, -result)
    assert_pytree_allclose(same_sign_result, result)


@pytest.mark.slow
def test_products_sign_covariance():
    """Test ProductsSignCovariance can be evaluated and is sign covariant."""
    _test_products_sign_covariance(3, True)
    _test_products_sign_covariance(1, False)
