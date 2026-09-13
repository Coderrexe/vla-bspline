#!/bin/bash
#SBATCH --job-name=calv_clock_audit
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=calv_clock_audit-%j.out
set -euo pipefail
SOURCE=$1
LOG_SOURCE=$2
MATRIX=$3
[[ "$MATRIX" =~ ^[0-9]+$ ]] || exit 2
LOGS=()
for ((index=0; index<12; index++)); do
  LOGS+=("$LOG_SOURCE/calv_clock_matrix-${MATRIX}_${index}.out")
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/zs442/.conda/envs/lerobot/bin/python \
  "$SOURCE/scripts/analysis/summarize_calvin_matched_timing.py" "${LOGS[@]}" \
  --out "/home/zs442/scratch/vla_bspline/artifacts/calvin_matched_timing_20260905/matrix_${MATRIX}_audit_${SLURM_JOB_ID}.json"
