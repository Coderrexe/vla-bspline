#!/bin/bash
#SBATCH --job-name=calv_clock_grouped
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --exclude=r817u29n05
#SBATCH --output=calv_clock_grouped-%A_%a.out
# Run all four decoders for one training seed on the same node. This removes
# node-to-node rendering as a possible source of mismatched initial requests.
set -euo pipefail
SOURCE=$1
NCHAINS=${2:-50}
SEED=$((1000 + SLURM_ARRAY_TASK_ID))
for ARM in A_native A_linear C_native C_retime; do
  bash "$SOURCE/cluster/eval_calvin_matched_timing.sbatch" \
    "$SOURCE" "$ARM" "$SEED" "$NCHAINS"
done
