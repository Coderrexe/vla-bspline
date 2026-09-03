#!/bin/bash
#SBATCH --job-name=rc_lang_audit
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"

SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_language_audit_20260821/audit_robocasa_curriculum_language.py"
BASE="$HOME/scratch/vla_bspline/robocasa_composite_seen"
GRANULAR="$HOME/scratch/vla_bspline/robocasa_composite_seen_gran"
OUT_DIR="$HOME/scratch/vla_bspline/outputs/robocasa_language_audit"
OUT="$OUT_DIR/easy4_${SLURM_JOB_ID}.json"

[[ -f "$SCRIPT" ]] || { echo "missing frozen audit source: $SCRIPT" >&2; exit 66; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite: $OUT" >&2; exit 73; }
mkdir -p "$OUT_DIR"
python "$SCRIPT" --base "$BASE" --granular "$GRANULAR" --out "$OUT"
test -s "$OUT"
sha256sum "$SCRIPT" "$OUT"
