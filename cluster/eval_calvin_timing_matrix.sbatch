#!/bin/bash
#SBATCH --job-name=calv_clock_matrix
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%4
#SBATCH --output=calv_clock_matrix-%A_%a.out
set -euo pipefail
SOURCE=$1
NCHAINS=${2:-50}
ARMS=(A_native A_linear C_native C_retime)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID % 4))]}
SEED=$((1000 + SLURM_ARRAY_TASK_ID / 4))
exec bash "$SOURCE/cluster/eval_calvin_matched_timing.sbatch" "$SOURCE" "$ARM" "$SEED" "$NCHAINS"
