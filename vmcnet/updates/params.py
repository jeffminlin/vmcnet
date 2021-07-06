"""Routines which handle model parameter updating."""
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import kfac_ferminet_alpha
from kfac_ferminet_alpha import utils as kfac_utils

import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.utils.typing import D, P, S


def create_grad_energy_update_param_fn(
    log_psi_apply: Callable[[P, jnp.ndarray], jnp.ndarray],
    local_energy_fn: Callable[[P, jnp.ndarray], jnp.ndarray],
    nchains: int,
    optimizer_apply: Callable[[P, P, S], Tuple[P, S]],
    get_position_fn: Callable[[D], jnp.ndarray],
    apply_pmap: bool = True,
) -> Callable[[D, P, S, jnp.ndarray], Tuple[P, S, Dict, jnp.ndarray]]:
    """Create the `update_param_fn` based on the gradient of the total energy.

    See :func:`~vmcnet.train.vmc.vmc_loop` for its usage.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        optimizer_apply (Callable): applies an update to the parameters. Has signature
            (grad_energy, params, optimizer_state) -> (new_params, new_optimizer_state).
        get_position_fn (Callable): gets the walker positions from the MCMC data
        apply_pmap (bool, optional): whether to apply jax.pmap to the walker function.
            If False, applies jax.jit. Defaults to True.

    Returns:
        Callable: function which updates the parameters given the current data, params,
        and optimizer state. The signature of this function is
            (data, params, optimizer_state, key)
            -> (new_params, new_optimizer_state, metrics, key)
        The function is pmapped if apply_pmap is True, and jitted if apply_pmap is
        False.
    """
    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply, local_energy_fn, nchains
    )

    def update_param_fn(data, params, optimizer_state, key):
        position = get_position_fn(data)
        energy_data, grad_energy = energy_data_val_and_grad(params, position)
        energy, aux_energy_data = energy_data

        grad_energy = utils.distribute.pmean_if_pmap(grad_energy)
        params, optimizer_state = optimizer_apply(grad_energy, params, optimizer_state)
        metrics = {"energy": energy, "variance": aux_energy_data[0]}
        return params, optimizer_state, metrics, key

    if not apply_pmap:
        return jax.jit(update_param_fn)

    pmapped_update_param_fn = utils.distribute.pmap(update_param_fn)

    def pmapped_update_param_fn_with_single_metrics(data, params, optimizer_state, key):
        params, optimizer_state, metrics, key = pmapped_update_param_fn(
            data, params, optimizer_state, key
        )
        metrics = utils.distribute.get_first(metrics)
        return params, optimizer_state, metrics, key

    return pmapped_update_param_fn_with_single_metrics


def create_kfac_update_param_fn(
    optimizer: kfac_ferminet_alpha.Optimizer,
    damping: jnp.float32,
    get_position_fn: Callable[[D], jnp.ndarray],
) -> Callable[[D, P, S, jnp.ndarray], Tuple[P, S, Dict, jnp.ndarray]]:
    """Create momentum-less KFAC update step function.

    Args:
        optimizer (kfac_ferminet_alpha.Optimizer): instance of the Optimizer class from
            kfac_ferminet_alpha
        damping (jnp.float32): damping coefficient
        get_position_from_data (Callable): function which gets the walker positions from
            the data. Has signature data -> jnp.ndarray

    Returns:
        Callable: function which updates the parameters given the current data, params,
        and optimizer state. The signature of this function is
            (data, params, optimizer_state, key)
            -> (new_params, new_optimizer_state, metrics, key)
    """
    momentum = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
    damping = kfac_utils.replicate_all_local_devices(jnp.asarray(damping))

    def update_param_fn(data, params, optimizer_state, key):
        key, subkey = utils.distribute.p_split(key)
        params, optimizer_state, stats = optimizer.step(
            params=params,
            state=optimizer_state,
            rng=subkey,
            data_iterator=iter([get_position_fn(data)]),
            momentum=momentum,
            damping=damping,
        )
        if optimizer.multi_device:
            energy = utils.distribute.get_first(stats["loss"])
            variance = utils.distribute.get_first(stats["aux"][0])
        metrics = {"energy": energy, "variance": variance}
        return params, optimizer_state, metrics, key

    return update_param_fn


def create_eval_update_param_fn(
    local_energy_fn: Callable[[P, jnp.ndarray], jnp.ndarray],
    nchains: int,
    get_position_fn: Callable[[D], jnp.ndarray],
):
    """No update/clipping/grad function which simply evaluates the local energies.

    Can be used to do simple unclipped MCMC with :func:`~vmcnet.train.vmc.vmc_loop`.

    Arguments:
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        get_position_fn (Callable): gets the walker positions from the MCMC data

    Returns:
        Callable: function which evaluates the local energies and averages them, without
        updating the parameters
    """

    def eval_update_param_fn(data, params, optimizer_state, key):
        local_energies = local_energy_fn(params, get_position_fn(data))
        energy = utils.distribute.mean_all_local_devices(local_energies)
        variance = (
            utils.distribute.mean_all_local_devices(jnp.square(local_energies - energy))
            * nchains
            / (nchains - 1)
        )  # adjust by n / (n - 1) to get an unbiased estimator
        metrics = {"energy": energy, "variance": variance}
        return params, optimizer_state, metrics, key

    return eval_update_param_fn
