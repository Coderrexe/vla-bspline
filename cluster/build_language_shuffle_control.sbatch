#!/bin/bash
#SBATCH --job-name=build_lang_shuffle
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# PROPOSAL ONLY — do not submit before reviewing the analyzer's eligible-frame
# coverage and lexical-overlap audit.  The output is a very small symlink overlay,
# not a copied dataset.

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SEED=${1:-1701}
case "$SEED" in
  ''|*[!0-9]*) echo "invalid seed: $SEED" >&2; exit 2 ;;
esac
# Set only after the analyzer report has been reviewed, for example:
#   sbatch --export=ALL,MIN_DERANGED_FRACTION=0.90 build_language_shuffle_control.sbatch 1701
MIN_DERANGED_FRACTION=${MIN_DERANGED_FRACTION:?set reviewed coverage threshold before building}

SCRIPT="$HOME/vla_bspline/cluster/build_language_shuffle_control.py"
SOURCE="$HOME/scratch/vla_bspline/libero_l10gran"
OUT="$HOME/scratch/vla_bspline/libero_l10gran_semshuffle_s${SEED}"
REPORT="$HOME/scratch/vla_bspline/outputs/language_dataset_audit/semshuffle_s${SEED}_${SLURM_JOB_ID}.json"

test -f "$SCRIPT"
test -d "$SOURCE/meta"
test ! -e "$OUT"
test ! -e "$REPORT"

python "$SCRIPT" analyze \
  --source "$SOURCE" --original-tasks 40 --seed "$SEED" --report "$REPORT"

# Fail closed against the coverage threshold pre-registered after the read-only
# design audit and before any policy outcome is observed.
REPORT="$REPORT" MIN_DERANGED_FRACTION="$MIN_DERANGED_FRACTION" python - <<'PY'
import json, os
p = json.load(open(os.environ["REPORT"]))
assert p["normalized_text_collision_count"] == 0, p["normalized_text_collision_count"]
observed = p["deranged_fraction_of_appended_frames"]
required = float(os.environ["MIN_DERANGED_FRACTION"])
assert 0.0 < required <= 1.0
assert observed >= required, (observed, required)
PY

python "$SCRIPT" build \
  --source "$SOURCE" --out "$OUT" --original-tasks 40 --seed "$SEED"

test -s "$OUT/meta/tasks.parquet"
test -s "$OUT/_semantic_shuffle_control/manifest.json"
echo "=== completed semantic-shuffle overlay job=${SLURM_JOB_ID} time=$(date -Is) ==="
