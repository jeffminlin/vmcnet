"""Test a hydrogen-like atom."""
import jax
import numpy as np
import vmcnet.examples.hydrogen_like_atom as hla
from vmcnet.mcmc.simple_position_amplitude import make_simple_position_amplitude_data

from .sgd_train import sgd_vmc_loop_with_logging


def test_hydrogen_like_vmc(caplog):
    """Test the wavefn exp(-a * r) converges (in 3-D) to a = nuclear charge."""
    # Problem parameters
    model_decay = 5.0
    nuclear_charge = 3.0
    ndim = 3

    # Training hyperparameters
    nchains = 100 * jax.local_device_count()
    nburn = 100
    nepochs = 10
    nsteps_per_param_update = 5
    std_move = 0.4
    learning_rate = 1e-2

    # Initialize model and chains of walkers
    log_psi_model = hla.HydrogenLikeWavefunction(model_decay)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    init_elec_pos = jax.random.normal(subkey, shape=(nchains, 1, ndim))

    key, subkey = jax.random.split(key)
    params = log_psi_model.init(key, init_elec_pos)
    amplitudes = log_psi_model.apply(params, init_elec_pos)
    data = make_simple_position_amplitude_data(init_elec_pos, amplitudes)

    # Local energy
    local_energy_fn = hla.make_hydrogen_like_local_energy(
        log_psi_model.apply, nuclear_charge, d=ndim
    )

    params = sgd_vmc_loop_with_logging(
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
        log_psi_model,
        local_energy_fn,
    )

    # Make sure the decay rate converged to the nuclear charge, since we're in 3-d
    np.testing.assert_allclose(jax.tree_leaves(params)[0], nuclear_charge, rtol=1e-6)
