import sys

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict, FieldReference
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.mcmc as mcmc
import vmcnet.physics as physics

dir = sys.argv[1]

import vmcnet.utils.io as io
import json
from ml_collections import ConfigDict

_, d, all_p, _, _ = io.reload_vmc_state(dir, "checkpoint.npz")

p_wf = all_p["wf"]


with open(dir + "/config.json", "r") as f:
    config = ConfigDict(json.load(f))

ion_pos = jnp.array(config.problem.ion_pos)
ion_charges = jnp.array(config.problem.ion_charges)
nelec = jnp.array(config.problem.nelec)

import vmcnet.models.construct as construct

slog_psi = construct.get_model_from_config(config.model, nelec, ion_pos, ion_charges)
log_psi_apply = construct.slog_psi_to_log_psi_apply(slog_psi.apply)

"""Get a default model configuration from a model type."""
orthogonal_init = {"type": "orthogonal", "scale": 1.0}
normal_init = {"type": "normal"}

# tie together the values of ferminet_backflow.cyclic_spins and
# invariance.cyclic_spins
cyclic_spins = FieldReference(False)

input_streams = {
    "include_2e_stream": True,
    "include_ei_norm": True,
    "include_ee_norm": True,
}

backflow_config = {
    "ndense_list": ((16, 4), (16, 4), (16, 4), (16,)),
    "kernel_init_unmixed": {"type": "orthogonal", "scale": 2.0},
    "kernel_init_mixed": orthogonal_init,
    "kernel_init_transformer": orthogonal_init,
    "kernel_init_2e_1e_stream": orthogonal_init,
    "kernel_init_2e_2e_stream": {"type": "orthogonal", "scale": 2.0},
    "bias_init_1e_stream": normal_init,
    "bias_init_2e_stream": normal_init,
    "bias_init_transformer": normal_init,
    "activation_fn": "tanh",
    "use_bias": True,
    "one_electron_skip": True,
    "one_electron_skip_scale": 1.0,
    "two_electron_skip": True,
    "two_electron_skip_scale": 1.0,
    "cyclic_spins": cyclic_spins,
    "use_transformer": False,
    "num_heads": 1,
}

surrogate_config = ConfigDict(
    {
        "input_streams": input_streams,
        "backflow": backflow_config,
    }
)

spin_split = construct.get_spin_split(nelec)

compute_input_streams = construct.get_compute_input_streams_from_config(
    surrogate_config.input_streams, ion_pos
)

backflow = construct.get_backflow_from_config(
    surrogate_config.backflow,
    spin_split,
)

from vmcnet.models.construct import FermiNetSurrogate

surrogate = FermiNetSurrogate(spin_split, compute_input_streams, backflow)
key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)

sg_apply = surrogate.apply

metrop_step_fn = dwpa.make_dynamic_pos_amp_gaussian_step(
    log_psi_apply,
    100,
    dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1),
)
burning_step = mcmc.metropolis.make_jitted_burning_step(
    metrop_step_fn, apply_pmap=False
)
walker_fn = mcmc.metropolis.make_jitted_walker_fn(10, metrop_step_fn, apply_pmap=False)


nelec_total = int(jnp.sum(nelec))

nchains = 100
key, position = physics.core.initialize_molecular_pos(
    key,
    nchains,
    ion_pos,
    ion_charges,
    nelec_total,
)

p_sg = surrogate.init(subkey, position[0:1])


amplitudes = log_psi_apply(p_wf, position)
data = dwpa.make_dynamic_width_position_amplitude_data(
    position,
    amplitudes,
    std_move=0.25,
    move_acceptance_sum=0.0,
    moves_since_update=0,
)


import vmcnet.physics.potential as potential

ei_term = potential.create_electron_ion_coulomb_potential(
    ion_pos, ion_charges, nparticles=1
)
ee_term = potential.create_electron_electron_coulomb_potential(nparticles=1)
import vmcnet.physics.random_particle as random_particle

kinetic_term = random_particle.create_random_particle_kinetic_energy(
    log_psi_apply, nparticles=1
)


def SPLE_apply(params, pos, perm):
    perm = jnp.array(perm)
    perm_pos = pos[..., perm, :]
    result = ei_term(params, perm_pos)
    result += ee_term(params, perm_pos)
    result += kinetic_term(params, pos, perm)
    return result


SPLE_apply = jax.jit(jax.vmap(SPLE_apply, in_axes=(None, 0, None), out_axes=0))


def msqe_loss(param_wf, param_sg, position):
    sg_predic = sg_apply(param_sg, position)
    sg_predic_1 = sg_predic[..., 0]

    sple_1 = SPLE_apply(param_wf, position, [0, 1, 2, 3])

    return jnp.mean((sg_predic_1 - sple_1) ** 2)


val_grad_msqe = jax.value_and_grad(msqe_loss, argnums=1)

print(f"# params in sg: {jax.flatten_util.ravel_pytree(p_sg)[0].shape[0]}")
print("Burning!")
data, key = mcmc.metropolis.burn_data(burning_step, 100, p_wf, data, key)

print("Learning!")
LR = 0.01

for i in range(1000):
    print(f"Epoch {i}")
    accept_ratio, data, key = walker_fn(p_wf, data, key)
    position = data["walker_data"]["position"]
    msqe, grad_p_sg = val_grad_msqe(p_wf, p_sg, position)
    p_sg = jax.tree_map(lambda x, y: x - LR * y, p_sg, grad_p_sg)
    print(f"MSQE: {msqe}")