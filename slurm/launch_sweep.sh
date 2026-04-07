#!/bin/bash
# Usage: ./slurm/launch_sweep.sh <entity/project/sweep_id> [num_agents=20]
# Example: ./slurm/launch_sweep.sh ggoldsh-university-of-california-berkeley/preconditioning/abc123 20
SWEEP_PATH=$1
N=${2:-20}

if [ -z "$SWEEP_PATH" ]; then
    echo "Usage: $0 <entity/project/sweep_id> [num_agents=20]"
    exit 1
fi

sbatch --array=0-$((N-1)) \
  --export=ALL,SWEEP_PATH="$SWEEP_PATH" \
  "$(dirname "$0")/sweep_agent.sh"
