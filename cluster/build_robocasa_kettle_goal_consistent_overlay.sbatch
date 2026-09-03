#!/bin/bash
# Build the label-only Kettle arm whose place and turn-on clauses name the
# same episode burner. No images, actions, states, rewards, or boundaries move.
#SBATCH --job-name=rc_kettle_goal_labels
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
PROJECT="$HOME/vla_bspline"
SOURCE="$HOME/scratch/vla_bspline/robocasa_composite_seen"
DEST="$HOME/scratch/vla_bspline/robocasa_composite_seen_official_kettle_goal_v1"
INPUTS="$HOME/scratch/vla_bspline/manifests/robocasa_official_kettle"
SCHEDULE="$INPUTS/robocasa_official_kettle_schedule.parquet"
SCHEDULE_MANIFEST="$INPUTS/robocasa_official_kettle_schedule_manifest.json"
SIGNATURES="$INPUTS/robocasa_official_kettle_action_signatures.json"
DERIVED_DIR="$HOME/scratch/vla_bspline/manifests/robocasa_official_kettle_goal"
DERIVED="$DERIVED_DIR/goal_consistent_schedule.parquet"
DERIVED_MANIFEST="$DERIVED_DIR/goal_consistent_schedule_manifest.json"
TRANSFORM="$PROJECT/scripts/data/build_robocasa_kettle_goal_consistent_schedule.py"
BUILDER="$PROJECT/scripts/data/build_robocasa_official_language_overlay.py"

for path in "$SCHEDULE" "$SCHEDULE_MANIFEST" "$SIGNATURES" "$TRANSFORM" "$BUILDER"; do
  test -s "$path" || exit 66
done
test ! -e "$DEST"; test ! -e "$DERIVED"; test ! -e "$DERIVED_MANIFEST"
mkdir -p "$DERIVED_DIR"
python "$TRANSFORM" --schedule "$SCHEDULE" --schedule_manifest "$SCHEDULE_MANIFEST" \
  --output "$DERIVED" --output_manifest "$DERIVED_MANIFEST"
python - "$DERIVED_MANIFEST" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["language_transform"] == "episode_burner_goal_copied_into_place_phase_v1"
assert d["episodes"] == 501 and d["frames"] == 228349
assert d["changed_instruction_rows"] == 58069
assert not d["physical_or_boundary_columns_changed"]
PY
python "$BUILDER" --source "$SOURCE" --destination "$DEST" --schedule "$DERIVED" \
  --schedule_manifest "$DERIVED_MANIFEST" --action_signatures "$SIGNATURES"
python - "$DEST/official_language_overlay_manifest.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["physical_action_stream_matches_official"]
assert d["episodes"] == 501 and d["frames"] == 228349
assert d["changed_column"] == "task_index"
assert any("place the kettle on the front-left" in s for s in d["added_task_strings"])
PY
sha256sum "$TRANSFORM" "$BUILDER" "$DERIVED" "$DERIVED_MANIFEST" \
  "$DEST/official_language_overlay_manifest.json"
