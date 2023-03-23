"""Spin calculations."""
from typing import Callable, List

import chex
import jax.numpy as jnp

from vmcnet.utils.distribute import get_mean_over_first_axis_fn
from vmcnet.utils.typing import Array, P, ModelApply, SLArray


def create_spin_square_expectation(
    local_spin_exchange: ModelApply[P], nelec: Array, nan_safe: bool = True
) -> Callable[[P, Array], chex.Numeric]:
    """Create a function which estimates the observable <psi | S^2 | psi> / <psi | psi>.

    We assume a wavefunction of spin-1/2 particles. If the wavefunction psi is given by

        psi(X) = antisymmetrize(F(R) xi(s)),

    where R is the real-space coordinates in X and s are the corresponding spins, and
    xi(s) is the spin configuration function corresponding to the S_z eigenstate with
    the first nelec[0] electrons being spin-up and the last nelec[1] electrons being
    spin-down.

    The S^2 operator can be written as (sum_i S_i)^2, which can be decomposed in terms
    of the total S_z operator and the raising and lowering operators S_{i+} and S_{i-},
    i.e. S_{i+} = S_{i,x} + iS_{i,y} and S_{j-} = S_{j,x} - iS_{j,y}.

    We have

        S^2 = S_z^2 - S_z + sum_i S_{i+} * S_{i-} + sum_{i != j} S_{i+} * S_{j-}.

    Direct calculation with the first three terms shows that

        <psi | S_z^2 - S_z + sum_i S_{i+} * S_{i-} | psi>
            = 0.25 (N_up - N_down)^2 + 0.5 (N_up + N_down).

    The remaining term is computed by integrating a local electron exchange term, as
    detailed in `create_local_spin_exchange`.

    Args:
        local_spin_exchange (Callable): a function which computes the local spin
            exchange term, F(R_{1 <-> 1 + nelec[0]}) / F(R). Must have the signature
            (params, x) -> local exchange term array with shape (x.shape[0],).
        nelec (Array): an array of size (2,) with nelec[0] being the number of spin-up
            electrons and nelec[1] being the number of spin-down electrons.
        nan_safe (bool, optional): flag which controls if jnp.nanmean is used instead of
            jnp.mean. Can be set to False when debugging if trying to find the source of
            unexpected nans. Defaults to True.

    Returns:
        Callable: a function which computes the S^2 expectation for a wavefunction (not
        necessarily normalized) given the number of spin-up and spin-down electrons and
        a collection of samples used to estimate the local spin exchange. Has the
        signature
        (params, x) -> <psi | S^2 | psi> / <psi | psi>
    """
    mean_fn = get_mean_over_first_axis_fn(nan_safe=nan_safe)

    def spin_square_expectation(params: P, x: Array) -> chex.Numeric:
        local_spin_exchange_out = local_spin_exchange(params, x)
        spin_square = (
            0.25 * (nelec[0] - nelec[1]) ** 2
            + 0.5 * (nelec[0] + nelec[1])
            + mean_fn(local_spin_exchange_out)
        )
        return spin_square

    return spin_square_expectation


def create_local_spin_exchange(
    slog_psi_apply: Callable[[P, Array], SLArray], nelec: List[int]
) -> ModelApply[P]:
    """Create the local observable from exchange of a spin-up and spin-down electron.

    We assume a wavefunction of spin-1/2 particles. If the wavefunction psi is given by

        psi(X) = antisymmetrize(F(R) xi(s)),

    where R is the real-space coordinates in X and s are the corresponding spins, and
    xi(s) is the spin configuration function corresponding to the S_z eigenstate with
    the first nelec[0] electrons being spin-up and the last nelec[1] electrons being
    spin-down.

    This factory creates a function which computes

        -1 * nelec[0] * nelec[1] * F(R_{1 <-> 1 + nelec[0]}) / F(R),

    where R_{1 <-> 1 + nelec[0]} denotes the exchange of the first spin-up and spin-down
    electron. When integrated over the distribution p(R) = |F(R)|^2 / int_R |F(R)|^2,
    this quantity gives the overlap integral

        <psi | sum_{i != j} S_{i+} * S_{j-} | psi> / <psi | psi>,

    where S_i+ and S_j- are the total spin raising and lowering operators, respectively,
    i.e. S_{i+} = S_{i,x} + iS_{i,y} and S_{j-} = S_{j,x} - iS_{j,y}.

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|
        nelec (Array): an array of size (2,) with nelec[0] being the number of spin-up
            electrons and nelec[1] being the number of spin-down electrons.

    Returns:
        Callable: function which computes a local spin exchange term, i.e. the function
        nelec[0] * nelec[1] * F(R_{1 <-> 1 + nelec[0]}) / F(R). Has signature
        (params, x) -> local exchange term array with shape (x.shape[0],)
    """
    if nelec[0] == 0 or nelec[1] == 0:
        # if only one spin species, then no exchange term
        def local_spin_exchange(params: P, x: Array) -> Array:
            return jnp.zeros_like(x[..., 0, 0])

    else:

        def local_spin_exchange(params: P, x: Array) -> Array:
            sign_psi, log_psi = slog_psi_apply(params, x)

            swapped_indices = list(range(x.shape[-2]))

            swapped_indices[0], swapped_indices[nelec[0]] = nelec[0], 0
            x_exchanged = jnp.take(x, jnp.array(swapped_indices), axis=-2)
            sign_exchanged_psi, log_exchanged_psi = slog_psi_apply(params, x_exchanged)

            sign_out = sign_psi * sign_exchanged_psi
            log_out = log_exchanged_psi - log_psi
            return -nelec[0] * nelec[1] * sign_out * jnp.exp(log_out)

    return local_spin_exchange
