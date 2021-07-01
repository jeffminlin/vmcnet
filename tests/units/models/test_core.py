"""Test core model building parts."""
import chex
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models


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
