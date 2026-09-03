#!/bin/bash
#SBATCH --job-name=t04_clause_build
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"

SNAP="$HOME/scratch/vla_bspline/code_snapshots/two_subgoal_clause_20260821_v2"
ROOT="$HOME/scratch/vla_bspline/hf_cache/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
ORIGINAL="$HOME/scratch/vla_bspline/libero_t04_exact_original_v1"
CLAUSE="$HOME/scratch/vla_bspline/libero_t04_exact_clause_v1"
SCRIPT="$SNAP/scripts/data/build_two_subgoal_clause_dataset.py"
test -s "$SCRIPT"; test -d "$ROOT/meta"; test -d "$ROOT/data"
test ! -e "$ORIGINAL"; test ! -e "$CLAUSE"

export PYTHONPATH="$SNAP"
python "$SNAP/tests/test_build_two_subgoal_clause_dataset.py" -v
python "$SCRIPT" --root "$ROOT" --original_out "$ORIGINAL" --clause_out "$CLAUSE" \
  --min_closed_run 48 --min_clause_frames 8

PROV="$CLAUSE/meta/two_subgoal_provenance.json"
MAP="$CLAUSE/meta/evaluator_dataset_task_mapping.json"
LABELS="$CLAUSE/meta/two_subgoal_labels.jsonl"
test -s "$PROV"; test -s "$MAP"; test -s "$LABELS"
PROV="$PROV" MAP="$MAP" LABELS="$LABELS" ROOT="$ROOT" ORIGINAL="$ORIGINAL" \
python - <<'PY'
import json,os
from pathlib import Path

p=json.load(open(os.environ["PROV"])); m=json.load(open(os.environ["MAP"]));
rows=[json.loads(line) for line in open(os.environ["LABELS"]) if line.strip()]
assert p["episode_count"]==71 and p["frame_count"]==19378
assert p["mapping_sha256"]==m["mapping_sha256"]
assert [(x["suite_task_id"],x["dataset_task_index"],x["episodes"],x["frames"])
        for x in m["targets"]]==[(0,5,33,9571),(4,0,38,9807)]
assert len(rows)==142
by_episode={}
for row in rows: by_episode.setdefault(row["episode_index"],[]).append(row)
assert len(by_episode)==71
for episode,parts in by_episode.items():
    parts=sorted(parts,key=lambda x:x["seg_start"])
    assert [x["clause_index"] for x in parts]==[0,1]
    assert parts[0]["seg_start"]==0
    assert parts[0]["seg_end"]==parts[1]["seg_start"]==parts[0]["switch_frame"]
    assert parts[1]["seg_end"]==parts[0]["frame_count"]
    assert parts[0]["label"]!=parts[1]["label"]
assert p["switches_per_episode"]==1 and p["full_frame_coverage"]
assert all(v["byte_identical"] for v in
           p["physical_non_language_verification"]["columns"].values())
root=Path(os.environ["ROOT"]); original=Path(os.environ["ORIGINAL"])
for source in root.glob("data/**/*.parquet"):
    target=original/source.relative_to(root)
    assert source.resolve().stat().st_ino==target.stat().st_ino
print("independent gate passed: 71 episodes, 19378 frames, one switch each")
PY
sha256sum "$SCRIPT" "$PROV" "$MAP" "$LABELS" \
  "$ORIGINAL/meta/paired_original_provenance.json"
echo "=== completed exact-task clause build job=$SLURM_JOB_ID ==="
