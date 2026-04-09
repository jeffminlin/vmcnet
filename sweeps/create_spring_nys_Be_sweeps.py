"""Create a wandb sweep for the spring_nys optimizer on the Beryllium atom.

The sweep runs a 42-trial grid search over learning_rate and damping with the
other hyperparameters fixed. Run this script once to register the sweep, then
run the agents using the printed commands.
"""

import copy
import wandb

ENTITY = "ggoldsh-university-of-california-berkeley"
PROJECT = "preconditioning"

BASE_SWEEP = {
    "program": "vmc-sweep",
    "command": [
        "${env}",
        "vmc-sweep",
        "--presets.name=Be",
        "--config.vmc.nepochs=20000",
        "--config.eval.nepochs=0",
        "--config.eval.nburn=0",
        "--config.logdir=/global/scratch/users/ggoldshlager/logs/sweeps/",
        "--config.wandb.mode=online",
        f"--config.wandb.project={PROJECT}",
        "--config.vmc.optimizer_type=spring_nys",
        "--config.vmc.optimizer.spring_nys.constrain_norm=False",
        "--config.vmc.check_for_nans=True",
    ],
    "method": "grid",
    "metric": {
        "goal": "minimize",
        "name": "variance_noclip_ema",
    },
    "parameters": {
        "vmc": {
            "parameters": {
                "optimizer": {
                    "parameters": {
                        "spring_nys": {
                            "parameters": {
                                # preconditioner_type filled in per sweep below
                                "learning_rate": {
                                    "values": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
                                },
                                "damping": {
                                    "values": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                                },
                            }
                        }
                    }
                },
            }
        }
    },
}


def main():
    config = copy.deepcopy(BASE_SWEEP)
    config["name"] = f"spring-nys-Be"

    sweep_id = wandb.sweep(config, entity=ENTITY, project=PROJECT)

    sweep_path = f"{ENTITY}/{PROJECT}/{sweep_id}"
    print("\nRun one trial:")
    print(f"  wandb agent --count 1 {sweep_path}")

    print("\nLaunch on cluster (42 agents, one per grid point):")
    print(f"  ./slurm/launch_sweep.sh {sweep_path} 42")


if __name__ == "__main__":
    main()