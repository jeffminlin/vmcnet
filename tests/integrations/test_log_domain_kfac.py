"""Test that KFAC treats Dense and LogDomainDense in the same way."""
import kfac_jax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kfac_jax import utils as kfac_utils

import vmcnet.models as models
import vmcnet.utils as utils
import vmcnet.utils.curvature_tags_and_blocks as curvature_tags_and_blocks
from vmcnet.utils.slog_helpers import array_to_slog, array_from_slog

from tests.test_utils import (
    assert_pytree_allclose,
    get_dense_and_log_domain_dense_same_params,
)


def _make_optimizer_from_loss_and_grad(loss_and_grad):
    return kfac_jax.Optimizer(
        loss_and_grad,
        l2_reg=0.0,
        norm_constraint=0.001,
        value_func_has_aux=False,
        learning_rate_schedule=lambda t: 1e-4,
        curvature_ema=0.95,
        inverse_update_period=1,
        min_damping=1e-4,
        num_burnin_steps=0,
        register_only_generic=False,
        estimation_mode="fisher_exact",
        multi_device=True,
        pmap_axis_name=utils.distribute.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
    )


@jax.pmap
def _data_iterator(key):
    return jax.random.uniform(key, (20, 1))


def _train(nsteps, loss_and_grad, model_params, batch, key, logdomain=False):
    """Train a model with KFAC and return params and loss for all steps."""
    # Distribute
    model_params = utils.distribute.replicate_all_local_devices(model_params)
    batch = utils.distribute.default_distribute_data(batch)
    key = utils.distribute.make_different_rng_key_on_all_devices(key)

    # Make a KFAC optimizer
    optimizer = _make_optimizer_from_loss_and_grad(loss_and_grad)

    key, subkey = utils.distribute.p_split(key)
    optimizer_state = optimizer.init(model_params, subkey, batch)

    # Train and record results for all steps
    momentum = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
    damping = kfac_utils.replicate_all_local_devices(jnp.asarray(0.001))

    training_results = []
    for _ in range(nsteps):
        key, subkey1 = utils.distribute.p_split(key)
        key, subkey2 = utils.distribute.p_split(key)
        new_params, new_optimizer_state, stats = optimizer.step(
            params=model_params,
            state=optimizer_state,
            rng=subkey1,
            data_iterator=iter([_data_iterator(subkey2)]),
            momentum=momentum,
            damping=damping,
        )
        loss = utils.distribute.get_first(stats["loss"])
        if logdomain:
            bias = new_params["params"]["kernel"][..., -1, :]
            kernel = new_params["params"]["kernel"][..., :-1, :]
        else:
            bias = new_params["params"]["bias"]
            kernel = new_params["params"]["kernel"]
        model_params, optimizer_state = new_params, new_optimizer_state

        training_results.append((np.array(kernel), np.array(bias), jnp.array(loss)))

    return training_results


@pytest.mark.slow
def test_log_domain_dense_kfac_matches_dense_kfac():
    """Test that KFAC trains LogDomainDense in the same way as Dense.

    Here a basic supervised multilinear regression problem is used.
    """
    nfeatures = 5

    def target_fn(x):
        """Target function is linear: f(x) = (2x, 3x, 4x, 5x, 6x)."""
        return jnp.dot(x, jnp.array([[2.0, 3.0, 4.0, 5.0, 6.0]]))

    # Make loss functions which should be equivalent
    dense_layer = models.core.Dense(nfeatures)
    logdomaindense_layer = models.core.LogDomainDense(nfeatures)

    def dense_loss(params, x):
        prediction = dense_layer.apply(params, x)
        target = target_fn(x)
        kfac_jax.register_squared_error_loss(prediction, target)
        return utils.distribute.mean_all_local_devices(jnp.square(prediction - target))

    def logdomaindense_loss(params, x):
        slog_out = logdomaindense_layer.apply(params, array_to_slog(x))
        prediction = array_from_slog(slog_out)
        target = target_fn(x)
        kfac_jax.register_squared_error_loss(prediction, target)
        return utils.distribute.mean_all_local_devices(jnp.square(prediction - target))

    dense_loss_and_grad = jax.value_and_grad(dense_loss, argnums=0)
    logdomaindense_loss_and_grad = jax.value_and_grad(logdomaindense_loss, argnums=0)

    # Initialize one set of initial params and an arbitrary initial batch
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    batch = jax.random.uniform(subkey, (20 * jax.local_device_count(), 1))

    key, subkey = jax.random.split(key)
    (
        dense_params,
        logdomaindense_params,
    ) = get_dense_and_log_domain_dense_same_params(subkey, batch, dense_layer)

    nsteps = 6

    key, subkey = jax.random.split(key)
    dense_results = _train(
        nsteps, dense_loss_and_grad, dense_params, batch, subkey, logdomain=False
    )
    logdomaindense_results = _train(
        nsteps,
        logdomaindense_loss_and_grad,
        logdomaindense_params,
        batch,
        subkey,
        logdomain=True,
    )

    assert_pytree_allclose(dense_results, logdomaindense_results, rtol=1e-6)
