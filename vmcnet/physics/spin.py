"""Spin calculations."""
from typing import Callable

import jax.numpy as jnp

from vmcnet.utils.distribute import get_mean_over_first_axis_fn
from vmcnet.utils.typing import Array, P, ModelApply, SLArray


def create_local_spin_hop(
    slog_psi_apply: Callable[[P, Array], SLArray], nelec: Array
) -> ModelApply[P]:
    """Create the local observable from exchange of a spin-up and spin-down electron.

    We assume a wavefunction of spin-1/2 particles. If the wavefunction psi is given by

        psi(X) = antisymmetrize(F(R) xi(s)),

    where R is the real-space coordinates in X and s are the corresponding spins, and
    xi(s) is the spin configuration function corresponding to the S_z eigenstate with
    the first nelec[0] electrons being spin-up and the last nelec[1] electrons being
    spin-down.

    This factory creates a function which computes

        F(R_{1 <-> 1 + nelec[0]}) / F(R),

    where R_{1 <-> 1 + nelec[0]} denotes the exchange of the first spin-up and spin-down
    electron. When integrated over the distribution p(R) = |F(R)|^2 / int_R |F(R)|^2,
    this quantity gives the overlap integral

        -(1/(N_up N_down)) <psi | sum_{i != j} S_{i+} * S_{j-} | psi> / <psi | psi>,

    where S_i+ and S_j- are the total spin raising and lowering operators, respectively
    and the * operator means a tensor contraction along the Pauli spin index, i.e.

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|
        nelec (Array): an array of size (2,) with nelec[0] being the number of spin-up
            electrons and nelec[1] being the number of spin-down electrons.

    Returns:
        Callable: function which computes the local kinetic energy for continuous
        problems (as opposed to discrete/lattice problems), i.e. -0.5 nabla^2 psi / psi.
        Evaluates on batches due to the jax.vmap call, so it has signature
        (params, x) -> kinetic energy array with shape (x.shape[0],)
    """
    if nelec[0] == 0 or nelec[1] == 0:
        # if only one spin species, then no hopping term
        def local_spin_hop(params: P, x: Array) -> Array:
            return jnp.zeros_like(x[..., 0, 0])

    else:

        def local_spin_hop(params: P, x: Array) -> Array:
            sign_psi, log_psi = slog_psi_apply(params, x)

            swapped_indices = list(range(x.shape[-2]))
            swapped_indices[0], swapped_indices[nelec[0]] = nelec[0], 0
            x_hopped = jnp.take(x, jnp.array(swapped_indices), axis=-2)
            sign_hopped_psi, log_hopped_psi = slog_psi_apply(params, x_hopped)

            sign_out = sign_psi * sign_hopped_psi
            log_out = log_hopped_psi - log_psi
            return sign_out * jnp.exp(log_out)

    return local_spin_hop


def create_spin_square_expectation(
    local_spin_hop: ModelApply[P], nelec: Array, nan_safe: bool = True
) -> Callable[[P, Array], jnp.float32]:
    """Create a function which estimates the observable <psi | S^2 | psi>."""
    mean_fn = get_mean_over_first_axis_fn(nan_safe=nan_safe)

    def spin_square_expectation(params: P, x: Array) -> jnp.float32:
        local_spin_hop_out = local_spin_hop(params, x)
        spin_square = (
            0.25 * (nelec[0] - nelec[1]) ** 2
            + 0.5 * (nelec[0] + nelec[1])
            - nelec[0] * nelec[1] * mean_fn(local_spin_hop_out)
        )
        return spin_square

    return spin_square_expectation
