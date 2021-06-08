"""Example FermiNet VMC script to learn the Boron atom with Adam."""
import logging

import jax
import jax.numpy as jnp
import optax
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils
import vmcnet.utils.io as io


def main():
    """Main routines."""
    logging.info("Starting!")
    reload_checkpoint_dir = "logs/B/ferminet/adam/checkpoints"
    checkpoint_file = "10.npz"
    log_dir = "logs/B/ferminet/adam/reload"

    """Main routines."""
    logging.info("Starting!")

    ion_pos = jnp.array([[0.0, 0.0, 0.0]])
    ion_charges = jnp.array([5.0])

    nchains = 10 * jax.local_device_count()
    nepochs = 10
    nburn_zero = 0
    nsteps_per_param_update = 10
    nmoves_per_width_update = 100
    dummy_std_move = 0.08
    dummy_epoch = 0
    learning_rate = 1e-4

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    init_pos = jax.random.normal(subkey, (nchains, 5, 3))

    kernel_init = models.weights.get_kernel_initializer("orthogonal")
    bias_init = models.weights.get_bias_initializer("normal")

    activation_fn = jnp.tanh
    log_psi = models.construct.SingleDeterminantFermiNet(
        (3,),
        ((64, 8), (64, 8), (64, 8)),
        kernel_init,
        kernel_init,
        kernel_init,
        kernel_init,
        kernel_init,
        kernel_init,
        kernel_init,
        bias_init,
        bias_init,
        bias_init,
        activation_fn,
        ion_pos=ion_pos,
        isotropic_decay=True,
    )

    key, subkey = jax.random.split(key)
    params = log_psi.init(subkey, init_pos)
    amplitudes = log_psi.apply(params, init_pos)
    data = dwpa.make_dynamic_width_position_amplitude_data(
        init_pos, amplitudes, dummy_std_move
    )

    # Setup metropolis step
    metrop_step_fn = dwpa.make_dynamic_pos_amp_gaussian_step(
        log_psi.apply,
        nmoves_per_width_update,
        dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1),
    )

    # Define parameter updates
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    def optimizer_apply(grad, params, optimizer_state):
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    kinetic_fn = physics.kinetic.create_continuous_kinetic_energy(log_psi.apply)
    ei_potential_fn = physics.potential.create_electron_ion_coulomb_potential(
        ion_pos, ion_charges
    )
    ee_potential_fn = physics.potential.create_electron_electron_coulomb_potential()
    ii_potential_fn = physics.potential.create_ion_ion_coulomb_potential(
        ion_pos, ion_charges
    )

    local_energy_fn = physics.core.combine_local_energy_terms(
        [kinetic_fn, ei_potential_fn, ee_potential_fn, ii_potential_fn]
    )

    update_param_fn = updates.params.create_grad_energy_update_param_fn(
        log_psi.apply,
        local_energy_fn,
        nchains,
        optimizer_apply,
        get_position_fn=pacore.get_position_from_data,
    )

    (epoch, data, params, optimizer_state, key) = io.reload_params(
        (dummy_epoch, data, params, optimizer_state, key),
        reload_checkpoint_dir,
        checkpoint_file,
    )

    # Distribute everything via jax.pmap
    (
        data,
        params,
        optimizer_state,
        key,
    ) = utils.distribute.distribute_reloaded_data((data, params, optimizer_state, key))

    params, optimizer_state, data = train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        nchains,
        nburn_zero,
        nepochs,
        nsteps_per_param_update,
        metrop_step_fn,
        update_param_fn,
        key,
        logdir=log_dir,
        checkpoint_every=100,
    )
    logging.info("Completed!")


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    main()
