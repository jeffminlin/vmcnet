"""Create configuration of hyperparameters."""
import os

from ml_collections import ConfigDict


def get_default_config() -> ConfigDict:
    """Make a default configuration (single det FermiNet on LiH)."""
    config = ConfigDict(
        {
            "problem": get_problem_config(),
            "model": get_model_config(),
            "vmc": get_vmc_config(),
            "eval": get_eval_config(),
            "logdir": os.path.join(
                os.curdir,  # this will be relative to the calling script
                "logs",
            ),
            # if current_datetime_subfolder=True, will log into a subfolder named
            # according to the datetime at start
            "save_to_current_datetime_subfolder": True,
            "logging_level": "WARNING",
            "dtype": "float32",
            "initial_seed": 0,
        }
    )
    return config


def choose_model_type_in_config(model_config):
    """Given a model config with a specified type, select the specified model."""
    model_type = model_config.type
    model_config = model_config[model_type]
    model_config.type = model_type
    return model_config


def get_model_config() -> ConfigDict:
    """Get a default model configuration from a model type."""
    orthogonal_init = ConfigDict({"type": "orthogonal", "scale": 1.0})
    normal_init = ConfigDict({"type": "normal"})
    ferminet_backflow = ConfigDict(
        {
            "ndense_list": ((256, 16), (256, 16), (256, 16), (256,)),
            "kernel_init_unmixed": {"type": "orthogonal", "scale": 2.0},
            "kernel_init_mixed": orthogonal_init,
            "kernel_init_2e_1e_stream": orthogonal_init,
            "kernel_init_2e_2e_stream": {"type": "orthogonal", "scale": 2.0},
            "bias_init_1e_stream": normal_init,
            "bias_init_2e_stream": normal_init,
            "activation_fn": "tanh",
            "include_2e_stream": True,
            "include_ei_norm": True,
            "include_ee_norm": True,
            "use_bias": True,
            "skip_connection": True,
            "cyclic_spins": False,
        }
    )
    config = ConfigDict(
        {
            "type": "ferminet",
            "ferminet": ConfigDict(
                {
                    "backflow": ferminet_backflow,
                    "ndeterminants": 1,
                    "kernel_init_orbital_linear": {"type": "orthogonal", "scale": 2.0},
                    "kernel_init_envelope_dim": {"type": "ones"},
                    "kernel_init_envelope_ion": {"type": "ones"},
                    "bias_init_orbital_linear": normal_init,
                    "orbitals_use_bias": True,
                    "isotropic_decay": True,
                }
            ),
            "brute_force_antisym": ConfigDict(
                {
                    "backflow": ferminet_backflow,
                    "antisym_type": "double",
                    "ndense_resnet": 64,
                    "nlayers_resnet": 2,
                    "kernel_init_resnet": {"type": "orthogonal", "scale": 2.0},
                    "kernel_init_jastrow": {"type": "ones"},
                    "bias_init_resnet": normal_init,
                    "activation_fn_resnet": "tanh",
                    "resnet_use_bias": True,
                }
            ),
        }
    )
    return config


def get_problem_config() -> ConfigDict:
    """Get a default molecular configuration (LiH)."""
    problem_config = ConfigDict(
        {
            "ion_pos": ((0.0, 0.0, -1.5069621), (0.0, 0.0, 1.5069621)),
            "ion_charges": (1.0, 3.0),
            "nelec": (2, 2),
        }
    )
    return problem_config


def get_vmc_config() -> ConfigDict:
    """Get a default VMC training configuration."""
    vmc_config = ConfigDict(
        {
            "nchains": 2000,
            "nepochs": 200000,
            "nburn": 5000,
            "nsteps_per_param_update": 10,
            "nmoves_per_width_update": 100,
            "std_move": 0.25,
            "clip_threshold": 5.0,
            "optimizer_type": "kfac",
            "schedule_type": "inverse_time",
            "learning_rate": 1e-4,
            "learning_decay_rate": 1e-4,
            "checkpoint_every": 5000,
            "best_checkpoint_every": 100,
            "checkpoint_dir": "checkpoints",
            "checkpoint_variance_scale": 10,
            "nhistory_max": 200,
        }
    )
    return vmc_config


def get_eval_config() -> ConfigDict:
    """Get a default evaluation configuration."""
    eval_config = ConfigDict(
        {
            "nchains": 2000,
            "nburn": 5000,
            "nepochs": 20000,
            "nsteps_per_energy_eval": 10,
            "nmoves_per_width_update": 100,
            "std_move": 0.25,
            # if use_data_from_training=True, nchains, nmoves_per_width_update, and
            # std_move are completely ignored, and the data output from training is
            # used as the initial positions instead
            "use_data_from_training": False,
        }
    )
    return eval_config
