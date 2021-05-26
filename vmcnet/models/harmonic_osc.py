"""Harmonic oscillator model."""
from dataclasses import InitVar
from typing import List, Sequence, Union

import flax
import jax
import jax.numpy as jnp


def make_hermite_polynomials(x: jnp.ndarray) -> jnp.ndarray:
    """Compute the first n Hermite polynomials evaluated at n points.

    Because the Hermite polynomials are orthogonal, they have a three-term recurrence
    given by

        H_n(x) = 2 * x * H_{n-1}(x) - 2 * (n - 1) * H_{n-2}(x).

    When there are n particles x_1, ..., x_n, this function simply evaluates the first n
    Hermite polynomials at these positions to form an nxn matrix with the particle index
    along the first axis and the polynomials along the second.

    Args:
        x (jnp.ndarray): an array of shape (..., nparticles, 1)

    Returns:
        jnp.ndarray: an array of shape (..., nparticles, nparticles), where the
        (..., i, j)th entry corresponds to H_j(x_i).
    """
    nparticles = x.shape[-2]
    if nparticles == 1:
        return jnp.ones_like(x)

    H = [jnp.ones_like(x), 2 * x]
    for n in range(2, nparticles):
        H.append(2 * x * H[n - 1] - 2 * (n - 1) * H[n - 2])

    return jnp.concatenate(H, axis=-1)


class HarmonicOscillatorOrbitals(flax.linen.Module):
    """Create orbitals for a set of non-interacting quantum harmonic oscillators.

    The single-particle quantum harmonic oscillator for particle i has the Hamiltonian

        H = -(1/2) d^2/dx^2 + (1/2) (omega * x)^2,

    where the first term is the second derivative/1-D Laplacian and the second term is
    the harmonic oscillator potential. The corresponding energy eigenfunctions take the
    simple form

        psi_n(x) = C * exp(-omega * x^2 / 2) * H_n(sqrt(omega) * x),

    where C is a normalizing constant and H_n is the nth Hermite polynomial. The
    corresponding energy level is simply E_n = n + (1/2).

    With N non-interacting particles, the total many-body Hamiltonian is simply
    H = sum_i H_i, and the above single-particle energy eigenfunctions become analogous
    to molecular orbitals. Due to the antisymmetry requirement on particle states, the
    N particles cannot occupy the same orbitals, so in the ground state configuration,
    the particles fill the lowest energy orbitals first. If some fixed spin
    configuration is specified, then the corresponding ground state configuration is the
    one where the particles fill the lowest energy orbitals per spin.

    This model evaluates the lowest n energy orbitals for n particles input; the .apply
    method has the signature (params, [x]) -> [orbital_matrix], where x has shape
    (..., n, 1) and orbital_matrix has shape (..., n, n), and the notation [**] is used
    to denote a pytree structure. The params consist of a model-specific omega. When
    this omega matches the omega in the potential, then this model.apply(params, x)
    evaluates the first n eigenfunctions of the Hamiltonian on x.

    Because the particles are completely non-interacting in this model, there are no
    interactions computed between the leaves of the input pytree.

    Attributes:
        omega_init (jnp.float32): initial value for omega in the model; when this
            matches the omega in the potential energy, then these orbitals become
            eigenfunctions
    """

    omega_init: jnp.float32

    @flax.linen.compact
    def _single_leaf_call(self, x: jnp.ndarray) -> jnp.ndarray:
        # x and omega * x have shape (..., n, 1)
        sqrt_omega_x = flax.linen.Dense(
            1,
            kernel_init=lambda key, shape, **kwargs: jnp.array(
                [[jnp.sqrt(self.omega_init)]]
            ),
            use_bias=False,
        )(x)

        # hermite matrices are (..., n, n)
        hermite_polys = make_hermite_polynomials(sqrt_omega_x)
        # envelopes are (..., n, 1)
        gaussian_envelopes = jnp.exp(-jnp.square(sqrt_omega_x) / 2.0)

        # orbitals are (..., n, n)
        orbitals = jax.tree_multimap(
            lambda x, y: x * y, gaussian_envelopes, hermite_polys
        )
        return orbitals

    def __call__(self, xs):
        return jax.tree_map(self._single_leaf_call, xs)
