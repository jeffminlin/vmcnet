"""Routines which handle model parameter updating."""
from typing import Callable, Dict, Tuple, TypeVar

import jax
import jax.numpy as jnp

import vmcnet.physics as physics
import vmcnet.utils as utils

D = TypeVar("D")  # represents MCMC data
P = TypeVar("P")  # Represents model parameters
S = TypeVar("S")  # represents optimizer state


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
        False. Because it is totally pure, the original (params, optimizer_state, key)
        buffers are deleted in the pmapped version via the `donate_argnums` argument so
        that XLA is potentially more memory-efficient on the GPU. See :func:`jax.pmap`.
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
