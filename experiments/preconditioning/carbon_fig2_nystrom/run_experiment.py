"""Run one Carbon minSR or smart EMA Nyström-minSR calculation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


TRAINING_ITERATIONS = 100_000
EVALUATION_ITERATIONS = 20_000
TRAINING_WALKERS = 1_000
EVALUATION_WALKERS = 2_000
LEARNING_RATE = 0.1
DAMPING = 1e-3
NYSTROM_RANK = 100
NYSTROM_EMA_DECAY = 0.999

RUN_NAMES = {
    "minsr": "carbon_minsr_seed0",
    "nystrom": "carbon_nystrom_r100_ema0999_seed0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(RUN_NAMES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two full-scale training steps and skip evaluation.",
    )
    return parser.parse_args()


def shared_config_args(output_root: Path, run_name: str, smoke: bool) -> list[str]:
    training_iterations = 2 if smoke else TRAINING_ITERATIONS
    evaluation_iterations = 0 if smoke else EVALUATION_ITERATIONS
    return [
        f"--config.notes=carbon_fig2_nystrom_{run_name}",
        f"--config.logdir={output_root}",
        "--config.save_to_current_datetime_subfolder=False",
        f"--config.subfolder_name={run_name}",
        "--config.initial_seed=0",
        "--config.wandb.mode=disabled",
        "--config.distribute=False",
        "--config.dtype=float32",
        "--config.problem.ion_pos=((0.0,0.0,0.0),)",
        "--config.problem.ion_charges=(6.0,)",
        "--config.problem.nelec=(4,2)",
        "--config.model.ferminet.ndeterminants=16",
        "--config.model.ferminet.full_det=True",
        f"--config.vmc.nchains={TRAINING_WALKERS}",
        f"--config.vmc.nepochs={training_iterations}",
        "--config.vmc.nburn=5000",
        "--config.vmc.nsteps_per_param_update=10",
        "--config.vmc.checkpoint_every=5000",
        "--config.vmc.best_checkpoint_every=100",
        "--config.vmc.clip_threshold=5.0",
        "--config.vmc.clip_center=mean",
        f"--config.eval.nchains={EVALUATION_WALKERS}",
        f"--config.eval.nepochs={evaluation_iterations}",
        "--config.eval.nburn=5000",
        "--config.eval.nsteps_per_param_update=10",
        "--config.eval.use_data_from_training=False",
    ]


def optimizer_config_args(method: str) -> list[str]:
    optimizer = "spring" if method == "minsr" else "spring_nystrom"
    prefix = f"--config.vmc.optimizer.{optimizer}"
    args = [
        f"--config.vmc.optimizer_type={optimizer}",
        f"{prefix}.schedule_type=inverse_time",
        f"{prefix}.learning_rate={LEARNING_RATE}",
        f"{prefix}.learning_decay_rate=1e-4",
        f"{prefix}.mu=0.0",
        f"{prefix}.constrain_norm=True",
        f"{prefix}.norm_constraint=0.001",
    ]
    if method == "minsr":
        args.append(f"{prefix}.damping={DAMPING}")
    else:
        args.extend(
            [
                f"{prefix}.sketch_damping={DAMPING}",
                f"{prefix}.nystrom_rank={NYSTROM_RANK}",
                f"{prefix}.nystrom_ema_decay={NYSTROM_EMA_DECAY}",
                f"{prefix}.metric_shift_strategy=regularization_coupled",
                f"{prefix}.metric_identity_shift=1.0",
                f"{prefix}.metric_normalization=none",
                f"{prefix}.nystrom_warmup_steps=0",
                f"{prefix}.nystrom_phasein_steps=0",
                f"{prefix}.eigenvalue_floor=1e-8",
                f"{prefix}.collect_during_warmup=True",
            ]
        )
    return args


def main() -> None:
    args = parse_args()
    executable = shutil.which("vmc-molecule")
    if executable is None:
        raise RuntimeError("vmc-molecule is not available on PATH")

    suffix = "_smoke" if args.smoke else ""
    run_name = RUN_NAMES[args.method] + suffix
    args.output_root.mkdir(parents=True, exist_ok=True)
    command = [executable]
    command.extend(shared_config_args(args.output_root, run_name, args.smoke))
    command.extend(optimizer_config_args(args.method))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
