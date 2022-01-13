"""Testing spin calcuations.

The spatial parts of the wavefunctions tested here are required to be antisymmetric
with respect to exchange of like-spin particles.
"""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.physics as physics
from vmcnet.utils.slog_helpers import array_to_slog


def _symmetric_spatial_wavefn(unused_params, x):
    del unused_params
    return jnp.ones_like(x[..., 0, 0]), jnp.zeros_like(x[..., 0, 0])


def _two_particle_antisymmetric_spatial_wavefn(unused_params, x):
    del unused_params
    wavefn_amp = x[..., 0, 0] - x[..., 1, 0]
    return array_to_slog(wavefn_amp)


def _four_particle_antisymmetric_spatial_wavefn(unused_params, x):
    del unused_params
    wavefn_amp = (
        (x[..., 0, 0] - x[..., 1, 0])
        * (x[..., 0, 0] - x[..., 2, 0])
        * (x[..., 0, 0] - x[..., 3, 0])
        * (x[..., 1, 0] - x[..., 2, 0])
        * (x[..., 1, 0] - x[..., 3, 0])
        * (x[..., 2, 0] - x[..., 3, 0])
    )
    return array_to_slog(wavefn_amp)


def _get_random_samples(seed, nelec_total):
    key = jax.random.PRNGKey(seed)
    nsamples = 10
    random_x = jax.random.normal(key, (nsamples, nelec_total, 3))
    return nsamples, random_x


def test_sz_zero_singlet_spin_overlap():
    """Compute overlap corresponding to exchange of a spin-up and spin-down electron.

    This test computes

        -(1/(N_up N_down)) <psi | sum_{i != j} S_{i+} * S_{j-} | psi> / <psi | psi>

    for a two-particle <S_z> = 0 spin singlet state, which should give 1.
    """
    # if spatial wavefunction is totally symmetric, then spin component is
    # antisymmetrized, so must be a singlet
    local_spin_exchange = physics.spin.create_local_spin_exchange(
        slog_psi_apply=_symmetric_spatial_wavefn, nelec=jnp.array([1, 1])
    )

    nsamples, random_x = _get_random_samples(seed=2, nelec_total=2)
    local_spin_exchange_out = local_spin_exchange(None, random_x)

    np.testing.assert_allclose(local_spin_exchange_out, jnp.ones((nsamples,)))


def test_sz_zero_triplet_spin_overlap():
    """Compute overlap corresponding to exchange of a spin-up and spin-down electron.

    This test computes

        -(1/(N_up N_down)) <psi | sum_{i != j} S_{i+} * S_{j-} | psi> / <psi | psi>

    for a two-particle <S_z> = 0 spin triplet state, which should give -1.
    """
    # if spatial wavefunction is antisymmetric, then spin component must be triplet
    local_spin_exchange = physics.spin.create_local_spin_exchange(
        slog_psi_apply=_two_particle_antisymmetric_spatial_wavefn,
        nelec=jnp.array([1, 1]),
    )

    nsamples, random_x = _get_random_samples(seed=53, nelec_total=2)
    local_spin_exchange_out = local_spin_exchange(None, random_x)

    np.testing.assert_allclose(local_spin_exchange_out, -jnp.ones((nsamples,)))


def test_sz_one_triplet_spin_overlap():
    """Checks for zero spin exchange overlap when only one spin species is present."""
    local_spin_exchange = physics.spin.create_local_spin_exchange(
        slog_psi_apply=_two_particle_antisymmetric_spatial_wavefn,
        nelec=jnp.array([2, 0]),
    )

    nsamples, random_x = _get_random_samples(seed=41, nelec_total=2)
    local_spin_exchange_out = local_spin_exchange(None, random_x)

    np.testing.assert_allclose(local_spin_exchange_out, jnp.zeros((nsamples,)))


def test_sz_zero_total_spin_six():
    """Compute <psi| S^2 | psi> = 6 with two up and two down elec.

    Direct computation shows that for a state with four electrons with n_up = 2 and
    n_down = 2, then <S^2> = 6 occurs when the spin component is completely symmetric,
    which implies that the spatial component is completely antisymmetric.
    """
    nelec = jnp.array([2, 2])
    local_spin_exchange = physics.spin.create_local_spin_exchange(
        slog_psi_apply=_four_particle_antisymmetric_spatial_wavefn, nelec=nelec
    )
    spin_square_expectation = physics.spin.create_spin_square_expectation(
        local_spin_exchange, nelec
    )

    _, random_x = _get_random_samples(seed=3, nelec_total=4)
    S2_expect = spin_square_expectation(None, random_x)

    np.testing.assert_allclose(S2_expect, 6)


def test_sz_one_total_spin_six():
    """Compute <psi| S^2 | psi> = 6 with three up and one down elec.

    Direct computation shows that for a state with four electrons with n_up = 3 and
    n_down = 1, then <S^2> = 6 occurs when the spin component is completely symmetric,
    which implies that the spatial component is completely antisymmetric.
    """
    nelec = jnp.array([3, 1])
    local_spin_exchange = physics.spin.create_local_spin_exchange(
        slog_psi_apply=_four_particle_antisymmetric_spatial_wavefn, nelec=nelec
    )
    spin_square_expectation = physics.spin.create_spin_square_expectation(
        local_spin_exchange, nelec
    )

    _, random_x = _get_random_samples(seed=3, nelec_total=4)
    S2_expect = spin_square_expectation(None, random_x)

    np.testing.assert_allclose(S2_expect, 6)


def test_sz_two_total_spin_six():
    """Compute <psi| S^2 | psi> = 6 with four up and zero down elec.

    Direct computation shows that for a state with four electrons with n_up = 4 and
    n_down = 0, then the state must be an <S^2> = 6 eigenstate (given a properly
    antisymmetrized spatial part).
    """
    nelec = jnp.array([4, 0])
    local_spin_exchange = physics.spin.create_local_spin_exchange(
        slog_psi_apply=_four_particle_antisymmetric_spatial_wavefn, nelec=nelec
    )
    spin_square_expectation = physics.spin.create_spin_square_expectation(
        local_spin_exchange, nelec
    )

    _, random_x = _get_random_samples(seed=5, nelec_total=4)
    S2_expect = spin_square_expectation(None, random_x)

    np.testing.assert_allclose(S2_expect, 6)
