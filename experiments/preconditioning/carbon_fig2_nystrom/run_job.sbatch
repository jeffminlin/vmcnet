#!/bin/bash
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

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
export PATH=/global/home/groups/co_esmath/ggoldshlager/envs/codex_preconditioning/bin:$PATH
export WANDB_MODE=disabled
export MPLCONFIGDIR=/global/scratch/users/ggoldshlager/codex/preconditioning/.matplotlib-cache

worktree=/global/home/groups/co_esmath/ggoldshlager/VMCNet_codex/preconditioning/vmcnet
output_root=/global/scratch/users/ggoldshlager/codex/preconditioning/carbon_fig2_nystrom

mkdir -p "$MPLCONFIGDIR" "$output_root"
cd "$worktree"

python experiments/preconditioning/carbon_fig2_nystrom/run_experiment.py \
  --method "$method" \
  --output-root "$output_root" \
  "${extra_args[@]}"
