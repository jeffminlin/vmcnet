"""Create one wandb sweep per spring_diag preconditioner type on the Carbon atom.

Each sweep runs a 20-trial Bayesian search over learning_rate and damping with the
preconditioner type fixed. Run this script once to register all 4 sweeps, then
launch agents individually:

    python sweeps/create_spring_diag_C_sweeps.py

    wandb agent --count 1 <entity>/preconditioning/<sweep_id>
"""

import copy
import wandb

ENTITY = "ggoldsh-university-of-california-berkeley"
PROJECT = "preconditioning"

PRECONDITIONER_TYPES = ["fisher", "march", "march_fisher", "ones"]

BASE_SWEEP = {
    "program": "vmc-sweep",
    "command": [
        "${env}",
        "vmc-sweep",
        "--presets.name=C",
        "--config.vmc.nepochs=100000",
        "--config.wandb.mode=online",
        f"--config.wandb.project={PROJECT}",
    ],
    "method": "bayes",
    "count": 20,
    "metric": {
        "goal": "minimize",
        "name": "energy_noclip_ema",
    },
    "parameters": {
        "vmc": {
            "parameters": {
                "optimizer_type": {"value": "spring_diag"},
                "optimizer": {
                    "parameters": {
                        "spring_diag": {
                            "parameters": {
                                # preconditioner_type filled in per sweep below
                                "learning_rate": {
                                    "distribution": "log_uniform_values",
                                    "min": 1.0e-3,
                                    "max": 1.0,
                                },
                                "damping": {
                                    "distribution": "log_uniform_values",
                                    "min": 1.0e-5,
                                    "max": 1.0,
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
    sweep_ids = {}
    for preconditioner in PRECONDITIONER_TYPES:
        config = copy.deepcopy(BASE_SWEEP)
        config["name"] = f"spring-diag-C-{preconditioner}"
        config["parameters"]["vmc"]["parameters"]["optimizer"]["parameters"][
            "spring_diag"
        ]["parameters"]["preconditioner_type"] = {"value": preconditioner}

        sweep_id = wandb.sweep(config, entity=ENTITY, project=PROJECT)
        sweep_ids[preconditioner] = sweep_id

    print("\nRun one trial per sweep:")
    for preconditioner, sweep_id in sweep_ids.items():
        sweep_path = f"{ENTITY}/{PROJECT}/{sweep_id}"
        print(f"  wandb agent --count 1 {sweep_path}  # {preconditioner}")

    print("\nLaunch on cluster (20 agents each):")
    for preconditioner, sweep_id in sweep_ids.items():
        sweep_path = f"{ENTITY}/{PROJECT}/{sweep_id}"
        print(f"  ./slurm/launch_sweep.sh {sweep_path}  # {preconditioner}")


if __name__ == "__main__":
    main()