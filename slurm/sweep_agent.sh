#!/bin/bash
#SBATCH --job-name=vmc-sweep
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=24:00:00
#SBATCH --output=./slurm_out/%A_%a_%x.out

module load python
module load cuda/10.2
conda activate vmcnet
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/global/software/sl-7.x86_64/modules/langs/cuda/10.2

# SWEEP_PATH is the full wandb path: entity/project/sweep_id
# e.g. ggoldsh-university-of-california-berkeley/preconditioning/abc123
wandb agent --count 1 "$SWEEP_PATH"
