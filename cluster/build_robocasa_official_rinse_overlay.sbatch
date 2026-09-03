#!/bin/bash
# Build the label-only official Rinse overlay after exact physical-stream gates.
#SBATCH --job-name=rc_official_rinse_overlay
#SBATCH --partition=devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"

SOURCE="$HOME/scratch/vla_bspline/robocasa_composite_seen"
DEST="$HOME/scratch/vla_bspline/robocasa_composite_seen_official_rinse_v1"
INPUTS="$HOME/scratch/vla_bspline/manifests/robocasa_official_rinse"
BUILDER="$HOME/vla_bspline/cluster/build_robocasa_official_language_overlay_v2.py"
for path in \
  "$BUILDER" \
  "$INPUTS/robocasa_official_RinseSinkBasin_schedule.parquet" \
  "$INPUTS/robocasa_official_RinseSinkBasin_schedule_manifest.json" \
  "$INPUTS/robocasa_official_RinseSinkBasin_action_signatures.json"; do
  [[ -f "$path" ]] || { echo "missing $path" >&2; exit 66; }
done
[[ ! -e "$DEST" ]] || { echo "refusing overwrite" >&2; exit 73; }
python "$BUILDER" \
  --source "$SOURCE" --destination "$DEST" \
  --schedule "$INPUTS/robocasa_official_RinseSinkBasin_schedule.parquet" \
  --schedule_manifest "$INPUTS/robocasa_official_RinseSinkBasin_schedule_manifest.json" \
  --action_signatures "$INPUTS/robocasa_official_RinseSinkBasin_action_signatures.json"
test -s "$DEST/official_language_overlay_manifest.json"
sha256sum "$BUILDER" "$DEST/official_language_overlay_manifest.json"
