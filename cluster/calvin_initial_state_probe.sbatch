#!/bin/bash
#SBATCH --job-name=calv_reset_probe
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=calv_reset_probe-%A_%a.out
set -euo pipefail
SOURCE=$1
BASE=/home/zs442/scratch/vla_bspline
RUN=${2:-$BASE/artifacts/calvin_initial_state_probe_20260907}
mkdir -p "$RUN"
PYTHONPATH= /home/zs442/.conda/envs/calvin/bin/python \
  "$SOURCE/scripts/eval/calvin_initial_state_probe.py" \
  --debug_root "$BASE/calvin_debug_dataset/validation" \
  --calvin_repo /home/zs442/vla_bspline/calvin --n_seq 50 \
  --run_index "$SLURM_ARRAY_TASK_ID" --out "$RUN/run_${SLURM_ARRAY_TASK_ID}.json"
