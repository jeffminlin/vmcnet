"""Integration tests for the quantum harmonic oscillator."""
import os
import shutil

import jax
import vmcnet.examples.harmonic_oscillator as qho
from vmcnet.mcmc.simple_position_amplitude import (
    make_simple_position_amplitude_data,
)
from sgd_train import sgd_vmc_loop_with_logging


def _make_initial_params_and_data(model_omega, nchains):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    random_particle_positions = jax.random.normal(subkey, shape=(nchains, 5, 1))

    # because there are 5 particles total, the spin split is (3, 2)
    log_psi_model = qho.make_harmonic_oscillator_spin_half_model(2, model_omega)

    key, subkey = jax.random.split(key)
    params = log_psi_model.init(subkey, random_particle_positions)
    amplitudes = log_psi_model.apply(params, random_particle_positions)
    return log_psi_model, params, random_particle_positions, amplitudes, key


def run(learning_rate_strat, nchains, learning_rate):
    """Test that the trainable sqrt(omega) converges to the true sqrt(spring constant).

    Integration test for the overall API, to make sure it comes together correctly and
    can optimize a simple 1 parameter model rapidly.
    """
    # Problem parameters
    model_omega = 2.5
    spring_constant = 1.5

    # Training hyperparameters
    nburn = 1000
    nepochs = 10000
    nsteps_per_param_update = 10
    std_move = 0.25

    dir = f"/Users/gil/PycharmProjects/VMCNet/vmcnet/tests/integrations/examples/convergence_data_nc_{nchains}_learning_{learning_rate_strat}_{learning_rate}"

    shutil.rmtree(
        dir,
        ignore_errors=True,
    )
    # Initialize model and chains of walkers
    (
        log_psi_model,
        params,
        random_particle_positions,
        amplitudes,
        key,
    ) = _make_initial_params_and_data(model_omega, nchains)
    data = make_simple_position_amplitude_data(random_particle_positions, amplitudes)

    # Local energy function
    local_energy_fn = qho.make_harmonic_oscillator_local_energy(
        spring_constant, log_psi_model.apply
    )

    _, params, _, _ = sgd_vmc_loop_with_logging(
        data,
        params,
        key,
        nchains,
        nburn,
        nepochs,
        nsteps_per_param_update,
        std_move,
        learning_rate,
        log_psi_model,
        local_energy_fn,
        logdir=dir,
        learning_rate_strat=learning_rate_strat,
    )


def main():
    learning_rate_strat = "decay"
    for learning_rate in [1e-2, 8e-2]:
        for nchains in [1000, 10]:
            run(learning_rate_strat, nchains, learning_rate)


if __name__ == "__main__":
    main()
