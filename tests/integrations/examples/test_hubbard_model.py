import jax
import numpy as np
import pytest

import vmcnet.examples.hubbard_model as hubbard
from vmcnet.mcmc.simple_position_amplitude import make_simple_position_amplitude_data

from .discrete_nolog_train import discrete_vmc_loop_with_logging


def _setup_hubbard_model():

    # Model parameters
    N=4
    N_up=2
    side_length=5

    # Training hyperparameters
    nchains = 100 * jax.local_device_count()
    nburn = 100
    nepochs = 100
    nsteps_per_param_update = 5
    std_move = 0.4
    learning_rate = 1.0

    # Initialize model and chains of walkers
    psi_model = hubbard.HubbardAnsatz(N,N_up,side_length)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    init_elec_pos = jax.random.randint(subkey, shape=(nchains, N), minval=0, maxval=side_length)

    key, subkey = jax.random.split(key)

    params = psi_model.init(key, init_elec_pos)
    amplitudes = psi_model.apply(params, init_elec_pos)
    data = make_simple_position_amplitude_data(init_elec_pos, amplitudes)

    # Local energy
    local_energy_fn = hubbard.make_hubbard_local_energy(psi_model.apply, side_length)

    return (
        params,
        nchains,
        nburn,
        nepochs,
        nsteps_per_param_update,
        std_move,
        learning_rate,
        psi_model,
        key,
        data,
        local_energy_fn,
        side_length,
    )


def test_hubbard_model_vmc(caplog):
    (
        params,
        nchains,
        nburn,
        nepochs,
        nsteps_per_param_update,
        std_move,
        learning_rate,
        psi_model,
        key,
        data,
        local_energy_fn,
        side_length,
    ) = _setup_hubbard_model()

    _, params, _, _ = discrete_vmc_loop_with_logging(
        caplog,
        data,
        params,
        key,
        nchains,
        nburn,
        nepochs,
        nsteps_per_param_update,
        std_move,
        learning_rate,
        psi_model,
        local_energy_fn,
        side_length,
    )

    np.testing.assert_allclose(0,0, rtol=1e-5) #not correct numbers yet


