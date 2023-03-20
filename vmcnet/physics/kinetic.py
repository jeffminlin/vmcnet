"""Kinetic energy terms."""
from typing import Callable

import jax

import vmcnet.physics as physics
from vmcnet.utils.typing import Array, P, ModelApply


def create_continuous_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array]
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

    def kinetic_energy_fn(params: P, x: Array) -> Array:
        return -0.5 * physics.core.laplacian_psi_over_psi(grad_log_psi_apply, params, x)

    return jax.vmap(kinetic_energy_fn, in_axes=(None, 0), out_axes=0)
