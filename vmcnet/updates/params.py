"""Routines which handle model parameter updating."""
from typing import Callable, Dict, Tuple, TypeVar

import jax
import jax.numpy as jnp

from vmcnet.updates.data import PositionAmplitudeData
import vmcnet.utils as utils

P = TypeVar("P")
O = TypeVar("O")  # represents optimizer state


def create_value_and_grad_energy_fn(
    log_psi_apply: Callable[[P, jnp.ndarray], jnp.ndarray],
    local_energy_fn: Callable[[P, jnp.ndarray], jnp.ndarray],
    nchains: int,
) -> Callable[
    [P, jnp.ndarray], Tuple[Tuple[jnp.float32, Tuple[jnp.float32, jnp.ndarray]], P]
]:
    """Create a function which computes unbiased energy gradients.

    Due to the Hermiticity of the Hamiltonian, we can get an unbiased lower variance
    estimate of the gradient of the expected energy than the naive gradient of the
    mean of sampled local energies. Specifically, the gradient of the expected energy
    expect[E_L] takes the form

        2 * expect[(E_L - expect[E_L]) * (grad_psi / psi)(x)],

    where E_L is the local energy and expect[] denotes the expectation with respect to
    the distribution |psi|^2.

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
        Callable: function which computes the energy value and gradient. Has signature
            (params, x)
            -> ((expected_energy, auxilliary_energy_data), grad_energy),
        where auxilliary_energy_data is the tuple (expected_variance, local_energies)
    """

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
    return energy_data_val_and_grad


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
    energy_data_val_and_grad = create_value_and_grad_energy_fn(
        log_psi_apply, local_energy_fn, nchains
    )

    def update_param_fn(data, params, optimizer_state):
        energy_data, grad_energy = energy_data_val_and_grad(params, data.position)
        energy, aux_energy_data = energy_data

        grad_energy = utils.distribute.pmean_if_pmap(grad_energy)
        params, optimizer_state = optimizer_apply(grad_energy, params, optimizer_state)
        metrics = {"energy": energy, "variance": aux_energy_data[0]}
        return params, optimizer_state, metrics

    return update_param_fn
