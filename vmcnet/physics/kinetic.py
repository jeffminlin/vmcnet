"""Kinetic energy terms."""
from typing import Callable, Union

import jax
import jax.numpy as jnp

import vmcnet.physics as physics
from vmcnet.utils.typing import P, ModelApply


def create_continuous_kinetic_energy(
    log_psi_apply: Callable[[P, jnp.ndarray], Union[jnp.float32, jnp.ndarray]]
) -> ModelApply[P]:
    """Create the local kinetic energy fn (params, x) -> -0.5 (nabla^2 psi(x) / psi(x)).

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|

    Returns:
        Callable: function which computes the local kinetic energy for continuous
        problems (as opposed to discrete/lattice problems), i.e. -0.5 nabla^2 psi / psi.
        Evaluates on batches due to the jax.vmap call, so it has signature
        (params, x) -> kinetic energy array with shape (x.shape[0],)
    """
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def kinetic_energy_fn(params: P, x: jnp.ndarray) -> jnp.float32:
        return -0.5 * physics.core.laplacian_psi_over_psi(grad_log_psi_apply, params, x)

    return jax.vmap(kinetic_energy_fn, in_axes=(None, 0), out_axes=0)


def create_hubbard_kinetic_energy(
    psi_apply: Callable[[P, jnp.ndarray], jnp.float32],
    side_length:int,
    ) -> ModelApply[P]:
    """Create the local kinetic energy fn (params, x) -> sum_{y~x}psi(y).
    Here y differs from x in one particle which has hopped to an adjacent site.

    Args:
        psi_apply (Callable): Has the signature (params, x) -> psi(x),
        side_length: side length of the finite lattice with periodic boundary condition

    Returns:
        Callable: function which computes the local kinetic energy.
        Evaluates on batches due to the jax.vmap call, so it has signature
        (params, x) -> kinetic energy array with shape (x.shape[0],)
    """

    def kinetic_energy_fn(params: P, x: jnp.ndarray) -> jnp.float32:
        return physics.core.adjacent_psi(psi_apply,params,side_length,x)/psi_apply(params,x)

    return jax.vmap(kinetic_energy_fn, in_axes=(None, 0), out_axes=0)
