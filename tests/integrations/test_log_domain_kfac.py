"""Test that KFAC treats Dense and LogDomainDense in the same way."""
import kfac_ferminet_alpha
import jax
import jax.numpy as jnp
from kfac_ferminet_alpha import utils as kfac_utils

import vmcnet.models as models
import vmcnet.utils as utils

from tests.test_utils import assert_pytree_allclose


def _make_optimizer_from_loss_and_grad(loss_and_grad):
    return kfac_ferminet_alpha.Optimizer(
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
    )


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
        kfac_ferminet_alpha.register_squared_error_loss(prediction, target)
        return utils.distribute.mean_all_local_devices(jnp.square(prediction - target))

    def logdomaindense_loss(params, x):
        sign_out, log_abs_out = logdomaindense_layer.apply(
            params, jnp.sign(x), jnp.log(jnp.abs(x))
        )
        prediction = sign_out * jnp.exp(log_abs_out)
        target = target_fn(x)
        kfac_ferminet_alpha.register_squared_error_loss(prediction, target)
        return utils.distribute.mean_all_local_devices(jnp.square(prediction - target))

    losses_and_grads = [
        jax.value_and_grad(loss, argnums=0)
        for loss in (dense_loss, logdomaindense_loss)
    ]

    # Initialize one set of initial params and an arbitrary initial batch
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    batch = jax.random.uniform(subkey, (20 * jax.local_device_count(), 1))

    key, subkey = jax.random.split(key)
    dense_params = dense_layer.init(subkey, batch)

    # Get concatenated dense kernel
    dense_bias = dense_params["params"]["bias"]
    dense_kernel = dense_params["params"]["kernel"]
    concat_dense = jnp.concatenate(
        [dense_kernel, jnp.expand_dims(dense_bias, 0)], axis=0
    )

    # Init logdomaindense params and replace the kernel with concatenated dense kernel
    logdomaindense_params = logdomaindense_layer.init(
        subkey, jnp.sign(batch), jnp.log(jnp.abs(batch))
    )
    logdomaindense_params = jax.tree_map(lambda _: concat_dense, logdomaindense_params)

    params = [dense_params, logdomaindense_params]

    # Distribute
    params = [
        utils.distribute.replicate_all_local_devices(model_params)
        for model_params in params
    ]
    batch = utils.distribute.default_distribute_data(batch)
    key = utils.distribute.make_different_rng_key_on_all_devices(key)

    # Make two kfac optimizers, both with the same keys and random inputs
    optimizers = [
        _make_optimizer_from_loss_and_grad(loss_and_grad)
        for loss_and_grad in losses_and_grads
    ]

    key, subkey = utils.distribute.p_split(key)
    optimizer_states = [
        optimizer.init(params[i], subkey, batch)
        for i, optimizer in enumerate(optimizers)
    ]

    # Train, checking that the outputs of the two optimizer steps are the same for each
    # iteration
    nsteps = 6
    momentum = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
    damping = kfac_utils.replicate_all_local_devices(jnp.asarray(0.001))

    @jax.pmap
    def data_iterator(key):
        return jax.random.uniform(key, (20, 1))

    for _ in range(nsteps):
        step_results = []
        key, subkey1 = utils.distribute.p_split(key)
        key, subkey2 = utils.distribute.p_split(key)
        for i, optimizer in enumerate(optimizers):
            new_params, new_optimizer_state, stats = optimizer.step(
                params=params[i],
                state=optimizer_states[i],
                rng=subkey1,
                data_iterator=iter([data_iterator(subkey2)]),
                momentum=momentum,
                damping=damping,
            )
            loss = utils.distribute.get_first(stats["loss"])
            if i == 0:
                bias = new_params["params"]["bias"]
                kernel = new_params["params"]["kernel"]
            elif i == 1:
                bias = new_params["params"]["kernel"][..., -1, :]
                kernel = new_params["params"]["kernel"][..., :-1, :]
            step_results.append((kernel, bias, loss))
            params[i], optimizer_states[i] = new_params, new_optimizer_state

        assert_pytree_allclose(step_results[0], step_results[1], rtol=1e-6)
