#!/bin/bash
#SBATCH --job-name=evlang_audit_t04
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
SUPPLEMENT="$HOME/scratch/vla_bspline/code_snapshots/event_language_t04_audit_20260821_v1"
BASE="$HOME/scratch/vla_bspline/code_snapshots/event_language_t04_20260821_v1"
WORK="$HOME/scratch/vla_bspline/event_language_t04_20260821_v1"
OUT="$HOME/scratch/vla_bspline/outputs/language_dataset_audit/event_t04_${SLURM_JOB_ID}.json"
export PYTHONPATH="$BASE"
test -s "$WORK/seg_labels.jsonl"
python "$SUPPLEMENT/scripts/analysis/audit_event_language_labels.py" \
  --labels "$WORK/seg_labels.jsonl" --out "$OUT" \
  --max_inversion_fraction 0.15 --max_gross_inversion_fraction 0.05 \
  --min_episodes_per_task 20
test -s "$OUT"
echo "=== completed label quality gate job=$SLURM_JOB_ID ==="
