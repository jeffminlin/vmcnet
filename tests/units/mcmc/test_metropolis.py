"""Test Metropolis routines."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.mcmc as mcmc
import vmcnet.updates as updates

from ..utils import make_dummy_data_params_and_key, make_dummy_metropolis_fn


def _dummy_model_apply(params, x):
    """Model eval that outputs indices of the flattened x in the shape of x."""
    return jnp.reshape(jnp.arange(jnp.size(x)), x.shape)


def test_metropolis_step():
    """Test the acceptance probability and data update for a single Metropolis step."""
    data, params, key = make_dummy_data_params_and_key()
    metrop_step_fn = make_dummy_metropolis_fn()

    accept_prob, new_data, _ = metrop_step_fn(data, params, key)

    np.testing.assert_allclose(accept_prob, 0.5)
    np.testing.assert_allclose(new_data, jnp.array([1, 0, 3, 0]))


def test_make_position_amplitude_gaussian_proposal():
    """Test that a model_apply is passed through correctly when making a proposal_fn."""
    proposal_fn = mcmc.metropolis.make_position_amplitude_gaussian_proposal(
        _dummy_model_apply, 1.0
    )
    positions, params, key = make_dummy_data_params_and_key()
    # use the "wrong" amplitudes here so we can make sure the "right" ones come out of
    # the proposal
    amplitudes = jnp.array([-1, -1, -1, -1])
    data = updates.data.PositionAmplitudeData(positions, amplitudes)

    new_data, _ = proposal_fn(params, data, key)
    np.testing.assert_allclose(new_data.amplitude, jnp.array([0, 1, 2, 3]))


def test_symmetric_acceptance():
    """Test acceptance probabilities in the symmetric proposal case."""
    old_amplitudes = jnp.array([1.0, -jnp.pi, 0.0, 1e-5])
    new_amplitudes = jnp.array([0.0, 1.0, 3.0, 4.0])

    log_old_amplitudes = jnp.log(jnp.abs(old_amplitudes))
    log_new_amplitudes = jnp.log(jnp.abs(new_amplitudes))

    expected_accept_prob = jnp.array([0.0, 1.0 / jnp.square(jnp.pi), 1.0, 1.0])

    acceptance_prob = mcmc.metropolis.metropolis_symmetric_acceptance(
        old_amplitudes, new_amplitudes, logabs=False
    )
    acceptance_prob_from_log = mcmc.metropolis.metropolis_symmetric_acceptance(
        log_old_amplitudes, log_new_amplitudes, logabs=True
    )

    np.testing.assert_allclose(acceptance_prob, expected_accept_prob)
    np.testing.assert_allclose(
        acceptance_prob_from_log, expected_accept_prob, rtol=1e-6
    )
