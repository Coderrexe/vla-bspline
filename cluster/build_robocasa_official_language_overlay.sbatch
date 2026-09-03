#!/bin/bash
# Build the label-only Kettle overlay from the audited official July-2026
# annotations. The 501 source demonstrations are physically byte-matched to
# combined episodes 1010..1510; only task_index/task vocabulary may change.
# Usage: sbatch build_robocasa_official_language_overlay.sbatch
#SBATCH --job-name=rc_official_overlay
#SBATCH --partition=devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"

PROJECT="$HOME/vla_bspline"
SOURCE="$HOME/scratch/vla_bspline/robocasa_composite_seen"
DEST="$HOME/scratch/vla_bspline/robocasa_composite_seen_official_kettle_v1"
INPUTS="$HOME/scratch/vla_bspline/manifests/robocasa_official_kettle"
BUILDER="$PROJECT/cluster/build_robocasa_official_language_overlay.py"
SCHEDULE="$INPUTS/robocasa_official_kettle_schedule.parquet"
SCHEDULE_MANIFEST="$INPUTS/robocasa_official_kettle_schedule_manifest.json"
SIGNATURES="$INPUTS/robocasa_official_kettle_action_signatures.json"

for path in "$BUILDER" "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SIGNATURES"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done
[[ -d "$SOURCE" ]] || { echo "missing source: $SOURCE" >&2; exit 66; }
[[ ! -e "$DEST" ]] || { echo "refusing overwrite: $DEST" >&2; exit 73; }

sha256sum "$BUILDER" "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SIGNATURES"
python "$BUILDER" \
  --source "$SOURCE" \
  --destination "$DEST" \
  --schedule "$SCHEDULE" \
  --schedule_manifest "$SCHEDULE_MANIFEST" \
  --action_signatures "$SIGNATURES"

test -s "$DEST/official_language_overlay_manifest.json"
sha256sum "$DEST/official_language_overlay_manifest.json"
