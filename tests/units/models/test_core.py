"""Test core model building parts."""
import jax
import jax.numpy as jnp
import pytest

import vmcnet.models as models
from vmcnet.utils.slog_helpers import array_to_slog
from vmcnet.utils.typing import Array, SLArray

from tests.test_utils import (
    get_dense_and_log_domain_dense_same_params,
    get_resnet_and_log_domain_resnet_same_params,
    assert_pytree_allclose,
)


@pytest.mark.slow
def test_dense_in_regular_and_log_domain_match():
    """Test that LogDomainDense does the same thing as Dense in the log domain."""
    nfeatures = 4
    dense_layer = models.core.Dense(nfeatures)
    log_domain_dense_layer = models.core.LogDomainDense(nfeatures)

    x = jnp.array([0.2, 3.0, 4.2, -2.3, 7.4, -3.0])  # random vector
    slog_x = array_to_slog(x)

    key = jax.random.PRNGKey(0)
    (
        dense_params,
        logdomaindense_params,
    ) = get_dense_and_log_domain_dense_same_params(key, x, dense_layer)

    out = dense_layer.apply(dense_params, x)
    slog_out = log_domain_dense_layer.apply(logdomaindense_params, slog_x)

    assert_pytree_allclose(slog_out, array_to_slog(out), rtol=1e-6)


@pytest.mark.slow
def test_resnet_in_regular_and_log_domain_match():
    """Test that LogDomainResnet does the same thing as SimpleResnet."""
    ninner = 4
    nfinal = 3
    nlayers = 5

    # Define activation function with simple analog in log domain
    def activation_fn(x: Array) -> Array:
        return jnp.sign(x) * (x**2) / 10

    # Define log domain version of the same activation function
    def log_domain_activation_fn(x: SLArray) -> SLArray:
        sign_x, log_x = x
        return (sign_x, 2 * log_x - jnp.log(10))

    resnet = models.core.SimpleResNet(
        ninner, nfinal, nlayers, activation_fn=activation_fn
    )
    log_domain_resnet = models.core.LogDomainResNet(
        ninner, nfinal, nlayers, activation_fn=log_domain_activation_fn
    )

    x = jnp.array([0.2, 3.0, 4.2, -2.3, 7.4, -3.0])  # random vector
    slog_x = array_to_slog(x)

    key = jax.random.PRNGKey(0)
    (
        resnet_params,
        log_domain_resnet_params,
    ) = get_resnet_and_log_domain_resnet_same_params(key, x, resnet)

    out = resnet.apply(resnet_params, x)
    slog_out = log_domain_resnet.apply(log_domain_resnet_params, slog_x)

    assert_pytree_allclose(slog_out, array_to_slog(out), rtol=1e-5)
