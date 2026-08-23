#!/bin/bash

set -euo pipefail

mode=${1:?usage: submit_jobs.sh smoke|full}
script_dir=$(cd "$(dirname "$0")" && pwd)
output_root=/pscratch/sd/g/ggoldsh/codex/preconditioning/carbon_fig2_nystrom
slurm_dir="$output_root/slurm"
mkdir -p "$slurm_dir"

case "$mode" in
  smoke)
    walltime=02:00:00
    suffix=smoke
    run_arg=--smoke
    ;;
  full)
    walltime=24:00:00
    suffix=full
    run_arg=
    ;;
  *)
    echo "mode must be smoke or full" >&2
    exit 2
    ;;
esac

for method in minsr nystrom; do
  sbatch \
    --job-name="c_${method}_${suffix}" \
    --time="$walltime" \
    --output="$slurm_dir/${method}_${suffix}_%j.out" \
    --error="$slurm_dir/${method}_${suffix}_%j.err" \
    "$script_dir/run_job.sbatch" "$method" "$run_arg"
done
