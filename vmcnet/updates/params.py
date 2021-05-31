"""Routines which handle model parameter updating."""
from typing import Callable, Dict, Tuple, TypeVar

import jax.numpy as jnp

from vmcnet.updates.data import PositionAmplitudeData
import vmcnet.physics as physics
import vmcnet.utils as utils

P = TypeVar("P")
O = TypeVar("O")  # represents optimizer state


def create_position_amplitude_data_update_param_fn(
    log_psi_apply: Callable[[P, jnp.ndarray], jnp.ndarray],
    local_energy_fn: Callable[[P, jnp.ndarray], jnp.ndarray],
    nchains: int,
    optimizer_apply: Callable[[P, P, O], Tuple[P, O]],
) -> Callable[[PositionAmplitudeData, P, O], Tuple[P, O, Dict]]:
    """Create the `update_param_fn` for PositionAmplitudeData.

    See :func:`~vmcnet.train.vmc.make_training_step` for its usage.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        optimizer_apply (Callable): applies an update to the parameters. Has signature
            (grad_energy, params, optimizer_state) -> (new_params, new_optimizer_state).

    Returns:
        Callable: function which updates the parameters given the current data, params,
        and optimizer state. The signature of this function is
            (PositionAmplitudeData, params, optimizer_state)
            -> (new_params, new_optimizer_state)
    """
    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply, local_energy_fn, nchains
    )

    def update_param_fn(data, params, optimizer_state):
        energy_data, grad_energy = energy_data_val_and_grad(params, data.position)
        energy, aux_energy_data = energy_data

        grad_energy = utils.distribute.pmean_if_pmap(grad_energy)
        params, optimizer_state = optimizer_apply(grad_energy, optimizer_state, params)
        metrics = {"energy": energy, "variance": aux_energy_data[0]}
        return params, optimizer_state, metrics

    return update_param_fn
