"""Harmonic oscillator model."""
import chex
import flax
import jax
import jax.numpy as jnp

import vmcnet.models as models
import vmcnet.physics as physics
from vmcnet.utils.typing import Array, LocalEnergyApply, ModelApply, P


def make_hermite_polynomials(x: Array) -> Array:
    """Compute the first n Hermite polynomials evaluated at n points.

    Because the Hermite polynomials are orthogonal, they have a three-term recurrence
    given by

        H_n(x) = 2 * x * H_{n-1}(x) - 2 * (n - 1) * H_{n-2}(x).

    When there are n particles x_1, ..., x_n, this function simply evaluates the first n
    Hermite polynomials at these positions to form an nxn matrix with the particle index
    along the first axis and the polynomials along the second.

    Args:
        x (Array): an array of shape (..., nparticles, 1)

    Returns:
        Array: an array of shape (..., nparticles, nparticles), where the
        (..., i, j)th entry corresponds to H_j(x_i).
    """
    nparticles = x.shape[-2]
    if nparticles == 1:
        return jnp.ones_like(x)

    H = [jnp.ones_like(x), 2 * x]
    for n in range(2, nparticles):
        H.append(2 * x * H[n - 1] - 2 * (n - 1) * H[n - 2])

    return jnp.concatenate(H, axis=-1)


class HarmonicOscillatorOrbitals(models.core.Module):
    """Create orbitals for a set of non-interacting quantum harmonic oscillators.

    The single-particle quantum harmonic oscillator for particle i has the Hamiltonian

        H = -(1/2) d^2/dx^2 + (1/2) (omega * x)^2,

    where the first term is the second derivative/1-D Laplacian and the second term is
    the harmonic oscillator potential. The corresponding energy eigenfunctions take the
    simple form

        psi_n(x) = C * exp(-omega * x^2 / 2) * H_n(sqrt(omega) * x),

    where C is a normalizing constant and H_n is the nth Hermite polynomial. The
    corresponding energy level is simply E_n = omega * (n + (1/2)).

    With N non-interacting particles, the total many-body Hamiltonian is simply
    H = sum_i H_i, and the above single-particle energy eigenfunctions become analogous
    to molecular orbitals. Due to the antisymmetry requirement on particle states, the
    N particles cannot occupy the same orbitals, so in the ground state configuration,
    the particles fill the lowest energy orbitals first. If some fixed spin
    configuration is specified, then the corresponding ground state configuration is the
    one where the particles fill the lowest energy orbitals per spin.

    For a single spin, the model evaluates the lowest n energy orbitals, where n is the
    number of particles for that spin. In this model, each spin corresponds to a leaf in
    the input pytree x.

    If x is a single array (spinless input), then when this omega matches the omega in
    the potential, then model.apply(params, x) evaluates the first n eigenfunctions
    of the Hamiltonian on x. If there are multiple leaves in x, each of which is a
    different spin, then this behavior is mapped over each leaf.

    In other words, the .apply method has the signature

        (params, [x]) -> [orbital_matrix],

    where x has shape (..., n, 1) and orbital_matrix has shape (..., n, n), and the
    bracket notation is used to denote a pytree structure of spin. For example, if the
    inputs are (params, {"up": x_up, "down": x_down}), then the output will be
    {"up": orbitals_up, "down": orbitals_down}. The params consist of a single number, a
    model-specific omega.

    Because the particles are completely non-interacting in this model, there are no
    interactions computed between the leaves of the input pytree (or even between
    particles in a single leaf).

    Attributes:
        omega_init (chex.Scalar): initial value for omega in the model; when this
            matches the omega in the potential energy, then these orbitals become
            eigenfunctions
    """

    omega_init: chex.Scalar

    @flax.linen.compact
    def _single_leaf_call(self, x: Array) -> Array:
        # x and omega * x have shape (..., n, 1)
        sqrt_omega_x = models.core.Dense(
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
        return gaussian_envelopes * hermite_polys

    def __call__(self, xs):
        """Compute the harmonic oscillator orbitals on each leaf of xs."""
        return jax.tree_map(self._single_leaf_call, xs)


def make_harmonic_oscillator_spin_half_model(
    nspin_first: int, model_omega_init: chex.Scalar
) -> models.core.Module:
    """Create a spin-1/2 quantum harmonic oscillator wavefunction (two spins).

    Args:
        nspin_first (int): number of the alpha spin type, where the number of spins of
            each type is (alpha, beta); the model assumes that its input along the
            second-to-last dimension has size alpha + beta
        model_omega_init (chex.Scalar): spring constant inside the model; when it
            matches the spring constant in the hamiltonian, then this model evaluates an
            eigenstate

    Returns:
        models.core.Module: spin-1/2 wavefunction for the quantum harmonic
        oscillator, with one trainable parameter (the model omega)
    """

    def split_spin_fn(x):
        return models.core.split(x, [nspin_first], axis=-2)

    orbitals = HarmonicOscillatorOrbitals(model_omega_init)
    logdet_fn = models.antisymmetry.logdet_product

    return models.core.ComposedModel([split_spin_fn, orbitals, logdet_fn])


def harmonic_oscillator_potential(omega: chex.Scalar, x: Array) -> chex.Numeric:
    """Potential energy for independent harmonic oscillators with spring constant omega.

    This function computes sum_i 0.5 * (omega * x_i)^2. If x has more than one axis,
    these are simply summed over, so this corresponds to an isotropic harmonic
    oscillator potential.

    This function should be vmapped in order to be applied to batches of inputs, as it
    expects the first axis of x to correspond to the particle index.

    Args:
        omega (chex.Scalar): spring constant
        x (Array): array of particle positions, where the first axis corresponds
            to particle index

    Returns:
        chex.Numeric: potential energy value for this configuration x
    """
    return 0.5 * jnp.sum(jnp.square(omega * x))


def make_harmonic_oscillator_local_energy(
    omega: chex.Scalar,
    log_psi_apply: ModelApply[P],
    local_energy_type: str = "standard",
) -> LocalEnergyApply[P]:
    """Factory to create a local energy fn for the harmonic oscillator log|psi|.

    Args:
        omega (chex.Scalar): spring constant for the harmonic oscillator
        log_psi_apply (Callable): function which evaluates log|psi| for a harmonic
            oscillator model wavefunction psi. Has the signature
            (params, x) -> log|psi(x)|.
        local_energy_type (bool): whether to use standard local energy, or integration
            by parts, or random particle local energy.

    Returns:
        Callable: local energy function with the signature (params, x) -> local energy
        associated to the wavefunction psi
    """

    def potential_fn(params: P, x: Array):
        # Note: harmonic oscillator potential is smooth, no need to apply sampling or
        # IBP here.
        del params
        return harmonic_oscillator_potential(omega, x)

    if local_energy_type == "random_particle":
        random_particle_kinetic_fn = (
            physics.random_particle.create_random_particle_kinetic_energy(
                log_psi_apply, nparticles=1
            )
        )
        return physics.random_particle.assemble_random_particle_local_energy(
            random_particle_kinetic_fn, [potential_fn], sample_kinetic=True
        )
    elif local_energy_type == "standard":
        kinetic_fn = physics.kinetic.create_laplacian_kinetic_energy(log_psi_apply)
    elif local_energy_type == "ibp":
        kinetic_fn = physics.ibp.create_gradient_squared_kinetic_energy(log_psi_apply)
    else:
        raise ValueError()

    return physics.core.combine_local_energy_terms([kinetic_fn, potential_fn])
