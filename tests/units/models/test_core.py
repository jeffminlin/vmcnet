"""Test core model building parts."""
import jax
import jax.numpy as jnp
import vmcnet.models as models
from vmcnet.utils.slog_helpers import array_to_slog
from tests.test_utils import (
    init_dense_and_logdomaindense_with_same_params,
    assert_pytree_allclose,
)


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
