#!/bin/bash
#SBATCH --job-name=audit_lang_shuffle
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Read-only gate for the semantic-label negative control. This computes exact
# derangeable coverage and lexical collisions but does not build an overlay.

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SEED=${1:-1701}
case "$SEED" in
  ''|*[!0-9]*) echo "invalid seed: $SEED" >&2; exit 2 ;;
esac
SCRIPT="$HOME/vla_bspline/cluster/build_language_shuffle_control.py"
SOURCE="$HOME/scratch/vla_bspline/libero_l10gran"
OUT="$HOME/scratch/vla_bspline/outputs/language_dataset_audit/semshuffle_analysis_s${SEED}_${SLURM_JOB_ID}.json"
test -s "$SCRIPT"
test -d "$SOURCE/meta"
test ! -e "$OUT"

python "$SCRIPT" analyze \
  --source "$SOURCE" \
  --original-tasks 40 \
  --seed "$SEED" \
  --report "$OUT"

test -s "$OUT"
OUT="$OUT" python - <<'PY'
import json
import os

result = json.load(open(os.environ["OUT"]))
assert result["control_kind"] == "same-support-caption-derangement"
assert result["appended_frame_count"] > 0
assert result["normalized_text_collision_count"] == 0
print(
    "coverage:", result["deranged_frame_count"], "/",
    result["appended_frame_count"],
    "frames; high-overlap pairs:", result["high_token_overlap_pair_count"],
)
PY
echo "=== completed shuffle audit job=${SLURM_JOB_ID} time=$(date -Is) ==="
