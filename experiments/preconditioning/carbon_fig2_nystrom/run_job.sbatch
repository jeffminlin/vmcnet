#!/bin/bash
#SBATCH --account=m1266_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm40g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1

set -euo pipefail

method=${1:?usage: run_job.sbatch METHOD [--smoke]}
mode=${2:-}
extra_args=()
if [[ -n "$mode" ]]; then
  if [[ "$mode" != "--smoke" ]]; then
    echo "second argument must be --smoke when provided" >&2
    exit 2
  fi
  extra_args+=("$mode")
fi

module purge
export PATH=/pscratch/sd/g/ggoldsh/envs/codex_preconditioning/bin:$PATH
export WANDB_MODE=disabled
export MPLCONFIGDIR=/pscratch/sd/g/ggoldsh/codex/preconditioning/.matplotlib-cache

worktree=/global/homes/g/ggoldsh/codex/preconditioning/vmcnet
output_root=/pscratch/sd/g/ggoldsh/codex/preconditioning/carbon_fig2_nystrom

mkdir -p "$MPLCONFIGDIR" "$output_root"
cd "$worktree"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun -n 1 -c 32 -G 1 --gpu-bind=none \
  python experiments/preconditioning/carbon_fig2_nystrom/run_experiment.py \
  --method "$method" \
  --output-root "$output_root" \
  "${extra_args[@]}"
