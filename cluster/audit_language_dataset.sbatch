#!/bin/bash
#SBATCH --job-name=lang_data_audit
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SCRIPT="$HOME/vla_bspline/cluster/audit_language_dataset.py"
ROOT="$HOME/scratch/vla_bspline/libero_l10gran"
OUT="$HOME/scratch/vla_bspline/outputs/language_dataset_audit/l10gran_${SLURM_JOB_ID}.json"
test -f "$SCRIPT"
test ! -e "$OUT"
python "$SCRIPT" --root "$ROOT" --original_tasks 40 --out "$OUT"
test -s "$OUT"
echo "=== completed language dataset audit job=${SLURM_JOB_ID} time=$(date -Is) ==="
