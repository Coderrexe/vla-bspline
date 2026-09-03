#!/bin/bash
# Same-allocation interim replication of the C original/clause pair. Keeping both
# treatments on one GPU UUID removes hardware assignment as a treatment confound.
# Claim-bearing H100 jobs remain queued independently.
#SBATCH --job-name=t04_Cpair_L40S
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err
set -euo pipefail

SEED=${1:-1001}
BASE="$HOME/vla_bspline/cluster/train_two_subgoal_clause_warm2x2.sbatch"
test -s "$BASE"
export TRAIN_GPU_CLASS=L40S
export RUN_SUFFIX=v1_l40s_samealloc

bash "$BASE" C original "$SEED" 7500
bash "$BASE" C clause "$SEED" 7500
