"""Test core model building parts."""
import chex
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models
from vmcnet.utils.slog_helpers import array_to_slog

from tests.test_utils import (
    init_dense_and_logdomaindense_with_same_params,
    assert_pytree_allclose,
)


def test_log_linear_exp_shape():
    """Test output shape of log linear exp."""
    signs = jnp.ones((5, 2, 4, 3))
    vals = jnp.zeros((5, 2, 4, 3))
    weights = jnp.ones((2, 7))

    out = models.core.log_linear_exp(signs, vals, weights, axis=-3)
    out_no_weights = models.core.log_linear_exp(signs, vals, axis=-3)

    desired_shape = (5, 7, 4, 3)
    desired_shape_no_weights = (5, 1, 4, 3)
    chex.assert_shape(out, desired_shape)
    chex.assert_shape(out_no_weights, desired_shape_no_weights)


def test_log_linear_exp_no_overflow():
    """Test that the log-linear-exp trick avoids overflow when any vals are big."""
    signs = jnp.array([-1.0, -1.0, 1.0, 1.0])
    vals = jnp.array([300.0, 100.0, 3000.0, 1.5])
    weights = jnp.reshape(jnp.array([-1.0, 2.0, 0.5, 0.6]), (4, 1))

    sign_out, log_out = models.core.log_linear_exp(signs, vals, weights, axis=0)

    # the output should be sign_out=1.0, log_out=log|0.5 * exp(3000) + tinier stuff|
    assert jnp.isfinite(log_out)
    np.testing.assert_allclose(sign_out, 1.0)
    np.testing.assert_allclose(log_out, jnp.log(0.5) + 3000.0)


def test_log_linear_exp_no_underflow():
    """Test that the log-linear-exp trick avoids underflow when all vals are small."""
    signs = jnp.array([-1.0, -1.0, 1.0, 1.0])
    vals = jnp.array([-4000.0, -5500.0, -3000.0, -1234.5])

    sign_out, log_out = models.core.log_linear_exp(signs, vals, axis=0)

    # the output should be sign_out=1.0, log_out=log|exp(-1234.5) + tinier stuff|
    np.testing.assert_allclose(sign_out, 1.0)
    np.testing.assert_allclose(log_out, -1234.5)


def test_log_linear_equals_log_linear_exp_log():
    """Test that log-linear-exp of sign(x), log|x| is just log-linear."""
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (9, 5))
    sign_x, log_abs_x = array_to_slog(x)

    key, subkey = jax.random.split(key)
    kernel = jax.random.normal(subkey, (5, 7))

    sign_linear_out, log_linear_out = array_to_slog(jnp.dot(x, kernel))
    sign_linear_exp_log_out, log_linear_exp_log_out = models.core.log_linear_exp(
        sign_x, log_abs_x, kernel, axis=-1
    )

    np.testing.assert_allclose(sign_linear_exp_log_out, sign_linear_out)
    np.testing.assert_allclose(log_linear_exp_log_out, log_linear_out, rtol=1e-5)


def test_dense_in_regular_and_log_domain_match():
    """Test that LogDomainDense does the same thing as Dense in the log domain."""
    nfeatures = 4
    dense_layer = models.core.Dense(nfeatures)
    logdomaindense_layer = models.core.LogDomainDense(nfeatures)

    x = jnp.array([0.2, 3.0, 4.2, -2.3, 7.4, -3.0])  # random vector
    slog_x = array_to_slog(x)

    key = jax.random.PRNGKey(0)
    (
        dense_params,
        logdomaindense_params,
    ) = init_dense_and_logdomaindense_with_same_params(
        key, x, dense_layer, logdomaindense_layer
    )
    out = dense_layer.apply(dense_params, x)
    slog_out = logdomaindense_layer.apply(logdomaindense_params, slog_x)
    assert_pytree_allclose(slog_out, array_to_slog(out), rtol=1e-6)
