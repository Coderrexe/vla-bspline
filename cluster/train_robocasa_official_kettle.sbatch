#!/bin/bash
# Matched Kettle language intervention:
#   original = one compound caption per demonstration
#   official = official per-frame pick/place/burner-specific phases
# Both consume exactly the same 501 physical demonstrations, initialize from
# the same corrected-contract checkpoint, and use identical optimization.
# Usage: sbatch train_robocasa_official_kettle.sbatch original|official|goal_consistent [seed] [steps]
#SBATCH --job-name=rc_official_kettle
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail

ARM=${1:?expected original, official, or goal_consistent}
SEED=${2:-1000}
STEPS=${3:-30000}
case "$ARM" in
  original) DATASET_NAME=robocasa_composite_seen ;;
  official) DATASET_NAME=robocasa_composite_seen_official_kettle_v1 ;;
  goal_consistent) DATASET_NAME=robocasa_composite_seen_official_kettle_goal_v1 ;;
  *) echo "unsupported arm: $ARM" >&2; exit 64 ;;
esac
for value in "$SEED" "$STEPS"; do
  case "$value" in ''|*[!0-9]*) echo "invalid integer: $value" >&2; exit 64 ;; esac
done

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
WORKERS=${SLURM_CPUS_PER_TASK:-24}
case "$WORKERS" in ''|*[!0-9]*) echo "invalid worker count: $WORKERS" >&2; exit 64 ;; esac

SNAP="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
export PYTHONPATH="$SNAP/src${PYTHONPATH:+:$PYTHONPATH}"
POLICY_DIR="$SNAP/src/lerobot/policies/smolvla_spline"
STATS_NAME=spline_stats_robocasa_composite_seen_all_v2_n8h24.json
DATASET="$HOME/scratch/vla_bspline/$DATASET_NAME"
DEFAULT_INIT="$HOME/scratch/vla_bspline/outputs/rccorrected_easy4_task_C_s1000_100000_2322332/checkpoints/100000/pretrained_model"
INIT=${INIT_OVERRIDE:-$DEFAULT_INIT}
OUT="$HOME/scratch/vla_bspline/outputs/rcofficial_kettle_${ARM}_C_s${SEED}_${STEPS}_${SLURM_JOB_ID}"
MANIFEST_DIR="$HOME/scratch/vla_bspline/manifests/robocasa_official_kettle_training"
MANIFEST="$MANIFEST_DIR/rcofficial_kettle_${ARM}_C_s${SEED}_${STEPS}_${SLURM_JOB_ID}.json"

for path in \
  "$POLICY_DIR/modeling_smolvla_spline.py" \
  "$POLICY_DIR/configuration_smolvla_spline.py" \
  "$POLICY_DIR/event_targets.py" \
  "$POLICY_DIR/stats_contract.py" \
  "$POLICY_DIR/$STATS_NAME" \
  "$INIT/model.safetensors" "$INIT/config.json" "$DATASET/meta/tasks.parquet"; do
  [[ -f "$path" ]] || { echo "missing immutable input: $path" >&2; exit 66; }
done
[[ ! -e "$OUT" && ! -e "$MANIFEST" ]] || { echo "refusing overwrite" >&2; exit 73; }
if [[ "$ARM" != original ]]; then
  test -s "$DATASET/official_language_overlay_manifest.json"
fi
mkdir -p "$MANIFEST_DIR"

EPISODES=$(python - <<'PY'
import json
episodes = list(range(1010, 1511))
assert len(episodes) == 501
print(json.dumps(episodes, separators=(",", ":")))
PY
)

python - "$MANIFEST" "$SNAP" "$DATASET" "$INIT" "$OUT" "$ARM" \
  "$SEED" "$STEPS" "$EPISODES" "$STATS_NAME" <<'PY'
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from lerobot.policies.smolvla_spline import event_targets, modeling_smolvla_spline, stats_contract

(
    manifest_path, snapshot, dataset, init, output, arm, seed, steps,
    episodes_json, stats_name,
) = sys.argv[1:]
episodes = json.loads(episodes_json)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

policy_dir = os.path.join(snapshot, "src/lerobot/policies/smolvla_spline")
source_files = {
    "modeling": os.path.join(policy_dir, "modeling_smolvla_spline.py"),
    "configuration": os.path.join(policy_dir, "configuration_smolvla_spline.py"),
    "event_targets": os.path.join(policy_dir, "event_targets.py"),
    "stats_contract": os.path.join(policy_dir, "stats_contract.py"),
    "corrected_stats": os.path.join(policy_dir, stats_name),
}
if os.path.realpath(inspect.getfile(modeling_smolvla_spline)) != os.path.realpath(source_files["modeling"]):
    raise RuntimeError("frozen spline source was not imported")
if sha256(source_files["corrected_stats"]) != "287be4152eb77bd9301a8d62e26123574356f9cb543e1b8050406e783ec78007":
    raise RuntimeError("corrected statistics hash mismatch")
if sha256(inspect.getfile(event_targets)) != "9335cae53e277aa76975d59f43c266b0a1beee17d12d7737fd2d09db3016e9f6":
    raise RuntimeError("event target source mismatch")

episode_meta = os.path.join(dataset, "meta/episodes/chunk-000/file-000.parquet")
task_meta = os.path.join(dataset, "meta/tasks.parquet")
rows = pq.read_table(episode_meta, columns=["episode_index", "source_prefix"]).to_pylist()
by_index = {int(row["episode_index"]): str(row["source_prefix"]) for row in rows}
if {by_index[index].split("/")[-2] for index in episodes} != {"KettleBoiling"}:
    raise RuntimeError("selected episodes are not exactly KettleBoiling")

columns = [
    "index", "episode_index", "frame_index", "timestamp", "action",
    "observation.state", "next.reward", "next.done",
]
dataset_view = ds.dataset(os.path.join(dataset, "data"), format="parquet")
selected = dataset_view.to_table(
    columns=columns + ["task_index"],
    filter=(ds.field("episode_index") >= 1010) & (ds.field("episode_index") <= 1510),
).sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])
if len(selected) != 228349:
    raise RuntimeError(f"expected 228349 Kettle frames, got {len(selected)}")
physical = hashlib.sha256()
language = hashlib.sha256()
for batch in selected.select(columns).to_batches(max_chunksize=32768):
    physical.update(batch.serialize().to_pybytes())
for batch in selected.select(["episode_index", "frame_index", "task_index"]).to_batches(max_chunksize=32768):
    language.update(batch.serialize().to_pybytes())

overlay_manifest = os.path.join(dataset, "official_language_overlay_manifest.json")
if arm in {"official", "goal_consistent"}:
    overlay = json.load(open(overlay_manifest, encoding="utf-8"))
    if not overlay.get("physical_action_stream_matches_official"):
        raise RuntimeError("official physical-stream gate is absent")
    overlay_hash = sha256(overlay_manifest)
else:
    overlay_hash = None

manifest = {
    "schema": "robocasa_official_kettle_train_v1",
    "job_id": os.environ["SLURM_JOB_ID"],
    "arm": arm,
    "seed": int(seed),
    "steps": int(steps),
    "episodes": len(episodes),
    "frames": len(selected),
    "episode_range": [1010, 1510],
    "initialization": os.path.realpath(init),
    "initial_model_sha256": sha256(os.path.join(init, "model.safetensors")),
    "initial_config_sha256": sha256(os.path.join(init, "config.json")),
    "dataset": os.path.realpath(dataset),
    "dataset_tasks_sha256": sha256(task_meta),
    "dataset_episode_metadata_sha256": sha256(episode_meta),
    "selected_physical_transition_sha256": physical.hexdigest(),
    "selected_language_schedule_sha256": language.hexdigest(),
    "official_overlay_manifest_sha256": overlay_hash,
    "source_snapshot": os.path.realpath(snapshot),
    "source_sha256": {name: sha256(path) for name, path in source_files.items()},
    "normalization_contract": {
        "stats_sha256": sha256(source_files["corrected_stats"]),
        "event_boundary_semantics": stats_contract.EVENT_BOUNDARY_SEMANTICS,
        "checkpoint_embedding_required": True,
    },
    "optimizer_semantics": "fresh optimizer/schedule from the manifest-bound initialization",
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip(),
    "output": os.path.realpath(output),
}
with open(manifest_path, "x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

python -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla_spline \
  --policy.n_ctrl=8 \
  --policy.min_seg=8 \
  --policy.horizon_max=24 \
  --policy.action_layout=robocasa12 \
  --policy.spline_stats_file_v2="$STATS_NAME" \
  --policy.predict_duration=true \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --policy.pretrained_path="$INIT" \
  --dataset.repo_id=ember-lab-berkeley/robocasa365-target-composite-seen \
  --dataset.root="$DATASET" \
  --dataset.episodes="$EPISODES" \
  --output_dir="$OUT" \
  --steps="$STEPS" \
  --batch_size=32 \
  --num_workers="$WORKERS" \
  --seed="$SEED" \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false

test -s "$OUT/checkpoints/last/pretrained_model/model.safetensors"
python - "$OUT" "$MANIFEST" <<'PY'
import json, os, shutil, sys
output, manifest = sys.argv[1:]
checkpoint = os.path.join(output, "checkpoints/last/pretrained_model")
config = json.load(open(os.path.join(checkpoint, "config.json"), encoding="utf-8"))
embedded = config.get("embedded_spline_stats")
if not embedded or embedded.get("boundary_semantics") != "production_event_index_k_exclusive":
    raise RuntimeError("saved checkpoint lacks corrected embedded stats")
shutil.copyfile(manifest, os.path.join(output, "launch_manifest.json"))
PY

echo "completed official-Kettle arm=$ARM seed=$SEED output=$OUT"
