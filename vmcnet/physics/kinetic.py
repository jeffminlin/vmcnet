"""Kinetic energy terms."""

from typing import Callable
import jax.numpy as jnp

from vmcnet.utils.typing import Array, P, ModelApply
from vmcnet.physics.fwdlap import zero_tangent_from_primal, lap  # type: ignore


def create_laplacian_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array],
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
        Evaluates on only a single configuration so must be externally vmapped
        to be applied to a batch of walkers.
    """

    def kinetic_energy_fn(params: P, x: Array) -> Array:
        """Compute -1/2 * (nabla^2 psi) / psi at x given a function which evaluates psi'(x)/psi.

        This function uses the identity

            (nabla^2 psi) / psi = nabla^2 log|psi|) + || nabla log|psi| ||^2

        to avoid leaving the log domain during the computation.

        This function should be vmapped in order to be applied to batches of inputs, as it
        completely flattens x in order to take second derivatives w.r.t. each component.
        """
        x_shape = x.shape
        flat_x = jnp.reshape(x, (-1,))
        n = flat_x.shape[0]
        eye = jnp.eye(n)

        def flattened_log_psi(flat_x_in):
            """Flattened input to flattened output version of log_psi."""
            return log_psi_apply(params, jnp.reshape(flat_x_in, x_shape))

        zero = zero_tangent_from_primal(flat_x)

        _, grads, laps = lap(flattened_log_psi, (flat_x,), (eye,), (zero,))
        laplacian_psi_over_psi = jnp.sum(grads**2) + laps

        return -0.5 * laplacian_psi_over_psi

    return kinetic_energy_fn
