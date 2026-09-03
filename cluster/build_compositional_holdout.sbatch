#!/bin/bash
# Build the t0+t1 training-only whole/clause pair and t0+t1-only spline stats.
# Usage: sbatch --nice=10000 build_compositional_holdout.sbatch SNAPSHOT_DIR
#SBATCH --job-name=comp_holdout_build
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
SNAP=${1:?immutable source snapshot directory}
ROOT="$HOME/scratch/vla_bspline/hf_cache/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
ORIGINAL="$HOME/scratch/vla_bspline/libero_comp_t01_original_v1"
CLAUSE="$HOME/scratch/vla_bspline/libero_comp_t01_clause_v1"
ARTIFACTS="$HOME/scratch/vla_bspline/artifacts/compositional_holdout_v1"
SPEC="$SNAP/configs/compositional_holdout_t01_to_t7.json"
BUILDER="$SNAP/scripts/data/build_two_subgoal_clause_dataset.py"
STATS_SCRIPT="$SNAP/scripts/data/gen_v2_stats_param.py"
STATS="$ARTIFACTS/spline_stats_libero_t01_v2_h24_n8.json"

test -s "$BUILDER"; test -s "$STATS_SCRIPT"; test -s "$SPEC"
test -d "$ROOT/meta"; test -d "$ROOT/data"
test ! -e "$ORIGINAL"; test ! -e "$CLAUSE"; test ! -e "$ARTIFACTS"
mkdir -p "$ARTIFACTS"

module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export PYTHONPATH="$SNAP:$SNAP/scripts/data:$SNAP/scripts/analysis"

python "$SNAP/tests/test_build_two_subgoal_clause_dataset.py" -v
python "$SNAP/tests/test_gen_v2_stats_param.py" -v
python "$BUILDER" \
  --root "$ROOT" \
  --original_out "$ORIGINAL" \
  --clause_out "$CLAUSE" \
  --target_spec "$SPEC" \
  --min_closed_run 48 \
  --min_clause_frames 8

PROV="$CLAUSE/meta/two_subgoal_provenance.json"
MAP="$CLAUSE/meta/evaluator_dataset_task_mapping.json"
LABELS="$CLAUSE/meta/two_subgoal_labels.jsonl"
test -s "$PROV"; test -s "$MAP"; test -s "$LABELS"

PROV="$PROV" MAP="$MAP" LABELS="$LABELS" SPEC="$SPEC" ROOT="$ROOT" \
ORIGINAL="$ORIGINAL" python - <<'PY'
import hashlib, json, os
from pathlib import Path

p=json.load(open(os.environ["PROV"])); m=json.load(open(os.environ["MAP"]));
rows=[json.loads(line) for line in open(os.environ["LABELS"]) if line.strip()]
expected=[(0,5,33,9571),(1,7,49,12702)]
actual=[(x["suite_task_id"],x["dataset_task_index"],x["episodes"],x["frames"])
        for x in m["targets"]]
assert actual==expected,(actual,expected)
assert p["episode_count"]==82 and p["frame_count"]==22273
assert len(rows)==164 and p["mapping_sha256"]==m["mapping_sha256"]
assert p["target_spec"]["sha256"]==hashlib.file_digest(
    open(os.environ["SPEC"],"rb"),"sha256").hexdigest()
selected={int(a["episode_index"]) for a in p["annotations"]}
held_out={7,9,25,29,30,41,63,74,82,83,96,98,124,135,148,160,161,163,
          171,174,188,195,196,205,208,221,222,223,237,246,250,256,265,
          266,275,281,286,289,293,297,318,331,373}
assert not selected & held_out
assert {int(a["suite_task_id"]) for a in p["annotations"]}=={0,1}
assert p["switches_per_episode"]==1 and p["full_frame_coverage"]
assert all(v["byte_identical"] for v in
           p["physical_non_language_verification"]["columns"].values())
assert set(p["selected_normalization_stats"])=={"paired_original","clause"}
for role,record in p["selected_normalization_stats"].items():
    assert record["row_count"]==22273,(role,record)
for dataset in (Path(os.environ["ORIGINAL"]),Path(os.environ["PROV"]).parents[1]):
    stats=json.load(open(dataset/"meta/stats.json"))
    for feature in ("action","observation.state","task_index"):
        assert stats[feature]["count"]==[22273],(dataset,feature,stats[feature]["count"])
root=Path(os.environ["ROOT"]); original=Path(os.environ["ORIGINAL"])
for source in root.glob("data/**/*.parquet"):
    target=original/source.relative_to(root)
    assert source.resolve().stat().st_ino==target.stat().st_ino
print("gate passed: 82 training episodes, 22,273 frames; all 43 t7 episodes held out")
PY

python "$STATS_SCRIPT" \
  --n_ctrl 8 --min_seg 8 --h_max 24 \
  --episodes_from_provenance "$PROV" \
  --out "$STATS"
STATS="$STATS" PROV="$PROV" python - <<'PY'
import hashlib,json,os
s=json.load(open(os.environ["STATS"])); p=json.load(open(os.environ["PROV"]))
expected=sorted(int(a["episode_index"]) for a in p["annotations"])
assert s["episode_selection"]["kind"]=="provenance_annotations"
assert s["episode_selection"]["episode_indices"]==expected
assert s["episode_selection"]["count"]==s["episode_selection"]["used_count"]==82
assert s["episode_selection"]["sha256"]==hashlib.file_digest(
    open(os.environ["PROV"],"rb"),"sha256").hexdigest()
assert s["n_ctrl"]==8 and s["min_seg"]==8 and s["h_max"]==24
print("stats gate passed:",s["n_anchors"],"anchors from exactly 82 training episodes")
PY

sha256sum "$BUILDER" "$STATS_SCRIPT" "$SPEC" "$PROV" "$MAP" "$LABELS" "$STATS" \
  > "$ARTIFACTS/SHA256SUMS"
sha256sum --check --strict "$ARTIFACTS/SHA256SUMS"
echo "=== completed compositional holdout data build job=$SLURM_JOB_ID ==="
