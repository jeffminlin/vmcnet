#!/bin/bash
#SBATCH --job-name=vmc-sweep
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#
#SBATCH --time=24:00:00
#SBATCH --output="/global/scratch/users/ggoldshlager/logs/sweeps/out"
#SBATCH --error="/global/scratch/users/ggoldshlager/logs/sweeps/error"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ggoldsh@berkeley.edu

source activate vmcnet

cd /global/home/groups/co_esmath/ggoldshlager/vmcnet

# SWEEP_PATH is the full wandb path: entity/project/sweep_id
# e.g. ggoldsh-university-of-california-berkeley/preconditioning/abc123
wandb agent --count 1 "$SWEEP_PATH"
