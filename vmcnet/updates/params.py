"""Routines which handle model parameter updating."""
from typing import Callable, Tuple, TypeVar

import jax
import jax.numpy as jnp

import vmcnet.utils as utils

P = TypeVar("P")
O = TypeVar("O")  # represents optimizer state


def create_position_amplitude_data_update_param_fn(
    log_psi_apply: Callable[[P, jnp.ndarray], jnp.ndarray],
    local_energy_fn: Callable[[P, jnp.ndarray], jnp.ndarray],
    nchains: int,
    optimizer_apply: Callable[[P, P, O], Tuple[P, O]],
):
    @jax.custom_jvp
    def compute_energy_data(params, positions):
        local_energies = local_energy_fn(params, positions)

        # TODO(Jeffmin) might be worth investigating the numerical stability of the XLA
        # compiled version of these two computations, since the quality of the gradients
        # is fairly crucial to the success of the algorithm
        energy = utils.distribute.pmean_if_pmap(jnp.mean(local_energies))
        variance = (
            utils.distribute.pmean_if_pmap(
                jnp.mean(jnp.square(local_energies - energy))
            )
            * nchains
            / (nchains - 1)
        )  # adjust by n / (n - 1) to get an unbiased estimator
        aux_data = (variance, local_energies)
        return energy, aux_data

    @compute_energy_data.defjvp
    def compute_energy_data_jvp(primals, tangents):
        params, positions = primals
        energy, aux_data = compute_energy_data(params, positions)
        _, local_energies = aux_data

        _, psi_tangents = jax.jvp(log_psi_apply, primals, tangents)
        primals_out = (energy, aux_data)
        tangents_out = (2.0 * jnp.dot(psi_tangents, local_energies - energy), aux_data)
        return primals_out, tangents_out

    energy_data_val_and_grad = jax.value_and_grad(
        compute_energy_data, argnums=0, has_aux=True
    )

    def update_param_fn(data, params, optimizer_state):
        energy_data, grad_energy = energy_data_val_and_grad(params, data.position)
        grad_energy = utils.distribute.pmean_if_pmap(grad_energy)
        params, optimizer_state = optimizer_apply(grad_energy, params, optimizer_state)
        metrics = {"energy": energy_data[0], "variance": energy_data[1][0]}
        return params, optimizer_state, metrics

    return update_param_fn
