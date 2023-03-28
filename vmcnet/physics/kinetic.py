"""Kinetic energy terms."""
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

import vmcnet.physics as physics
from vmcnet.utils.typing import Array, P, ModelApply


def create_continuous_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array],
    ion_pos,
    ibp=False,
    alpha=0.2,
    beta=0.1,
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

    # Evaluates fk(rk) for each k
    def f(x, ion_pos):
        # (n,m)
        dists = jnp.linalg.norm(
            jnp.expand_dims(x, -2) - jnp.expand_dims(ion_pos, -3), axis=-1
        )
        # (n,)
        min_dists = jnp.min(dists, axis=-1)

        return (erf((min_dists - alpha) / beta) + 1) / 2

    def sumf(x, ion_pos):
        return jnp.sum(f(x, ion_pos))

    gradf = jax.grad(sumf, argnums=0)

    def kinetic_energy_fn(params: P, x: Array) -> Array:
        if not ibp:
            return -0.5 * physics.core.laplacian_psi_over_psi(
                grad_log_psi_apply, params, x
            )

        # (n,1)
        fs = jnp.expand_dims(f(x, ion_pos), -1)
        # (n,1)
        gradfs = gradf(x, ion_pos)
        # (n,3)
        gradlogpsi = grad_log_psi_apply(params, x)

        return 0.5 * (
            jnp.sum(gradfs * gradlogpsi)
            + jnp.sum((gradlogpsi**2) * fs)
            - physics.core.laplacian_psi_over_psi(
                grad_log_psi_apply, params, x, weights=1 - fs
            )
        )

    return kinetic_energy_fn
