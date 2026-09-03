#!/bin/bash
# Fresh-optimizer, task-focused RoboCasa recovery stage from an immutable
# curriculum checkpoint.  The parent curriculum keeps running as the low-LR
# control; this branch resets the cosine schedule to test whether per-task
# exposure, rather than representation, is the remaining bottleneck.
#
# Usage:
#   sbatch train_robocasa_task_focus.sbatch \
#     MODE TASK BASE_RUN BASE_STEP [SEED] [STEPS]
# MODE: task, gran, or gran_pure. ``gran_pure`` is currently Kettle-only and
# selects the 251 episodes whose every frame uses one of the two clause labels,
# removing the 250 full-caption control episodes from the mixed overlay.
#SBATCH --job-name=rc_focus
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

MODE=${1:?task or gran required}
TASK=${2:?RoboCasa task required}
BASE_RUN=${3:?base curriculum run directory required}
BASE_STEP=${4:?base checkpoint step required}
SEED=${5:-1000}
STEPS=${6:-30000}
case "$MODE" in
  task) DATASET_NAME=robocasa_composite_seen ;;
  gran) DATASET_NAME=robocasa_composite_seen_gran ;;
  gran_pure) DATASET_NAME=robocasa_composite_seen_gran ;;
  *) echo "unsupported mode: $MODE" >&2; exit 64 ;;
esac
if [[ "$MODE" == gran_pure && "$TASK" != KettleBoiling ]]; then
  echo "gran_pure is currently provenance-locked to KettleBoiling" >&2
  exit 64
fi
case "$TASK" in
  KettleBoiling) START=1010; END=1510; EXPECTED=501 ;;
  RinseSinkBasin) START=3528; END=4036; EXPECTED=509 ;;
  ScrubCuttingBoard) START=4037; END=4540; EXPECTED=504 ;;
  StackBowlsCabinet) START=5546; END=6060; EXPECTED=515 ;;
  *) echo "unsupported task: $TASK" >&2; exit 64 ;;
esac
for value in "$BASE_STEP" "$SEED" "$STEPS"; do
  case "$value" in ''|*[!0-9]*) echo "invalid integer: $value" >&2; exit 64 ;; esac
done
printf -v STEP_DIR '%06d' "$BASE_STEP"

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

DATASET="$HOME/scratch/vla_bspline/$DATASET_NAME"
BASE_CKPT="$BASE_RUN/checkpoints/$STEP_DIR/pretrained_model"
TRAIN_STATE="$BASE_RUN/checkpoints/$STEP_DIR/training_state/training_step.json"
OUT="$HOME/scratch/vla_bspline/outputs/rcfocus_${TASK}_${MODE}_from${STEP_DIR}_s${SEED}_${STEPS}_${SLURM_JOB_ID}"
MANIFEST_DIR="$HOME/scratch/vla_bspline/manifests/robocasa_task_focus"
MANIFEST="$MANIFEST_DIR/rcfocus_${TASK}_${MODE}_from${STEP_DIR}_s${SEED}_${STEPS}_${SLURM_JOB_ID}.json"

[[ -d "$DATASET" ]] || { echo "missing dataset: $DATASET" >&2; exit 66; }
[[ -f "$BASE_CKPT/model.safetensors" ]] || { echo "missing checkpoint: $BASE_CKPT" >&2; exit 66; }
[[ -f "$TRAIN_STATE" ]] || { echo "missing training state: $TRAIN_STATE" >&2; exit 66; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite: $OUT" >&2; exit 73; }
[[ ! -e "$MANIFEST" ]] || { echo "refusing to overwrite: $MANIFEST" >&2; exit 73; }
mkdir -p "$MANIFEST_DIR"

EPISODES=$(START="$START" END="$END" MODE="$MODE" DATASET="$DATASET" python - <<'PY'
import json, os
import pyarrow.dataset as ds

start, end = int(os.environ["START"]), int(os.environ["END"])
if os.environ["MODE"] == "gran_pure":
    dataset = ds.dataset(os.path.join(os.environ["DATASET"], "data"), format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "task_index"],
        filter=(ds.field("episode_index") >= start) & (ds.field("episode_index") <= end),
    )
    rows = table.to_pylist()
    by_episode = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_index"]), set()).add(int(row["task_index"]))
    # Kettle clause indices are immutable in the audited granular metadata.
    episodes = sorted(
        episode for episode, labels in by_episode.items()
        if labels.issubset({384, 385}) and labels == {384, 385}
    )
    if len(episodes) != 251:
        raise RuntimeError(f"expected 251 pure-granular Kettle episodes, got {len(episodes)}")
else:
    episodes = list(range(start, end + 1))
print(json.dumps(episodes, separators=(",", ":")))
PY
)

python - "$MANIFEST" "$DATASET" "$BASE_CKPT" "$TRAIN_STATE" "$MODE" "$TASK" "$START" "$END" "$EXPECTED" "$SEED" "$STEPS" "$BASE_STEP" "$EPISODES" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys

import pyarrow.parquet as pq

(
    manifest_path, dataset, checkpoint, training_state_path, mode, task,
    start, end, expected, seed, steps, base_step, episodes_json,
) = sys.argv[1:]
start, end, expected = int(start), int(end), int(expected)
training_episodes = json.loads(episodes_json)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

state = json.load(open(training_state_path, encoding="utf-8"))
observed_step = int(state.get("step", state.get("training_step")))
if observed_step != int(base_step):
    raise RuntimeError(f"base step mismatch: {observed_step} != {base_step}")

episode_meta = os.path.join(dataset, "meta/episodes/chunk-000/file-000.parquet")
task_meta = os.path.join(dataset, "meta/tasks.parquet")
dataset_stats = os.path.join(dataset, "meta/stats.json")
rows = pq.read_table(episode_meta, columns=["episode_index", "source_prefix"]).to_pylist()
by_index = {int(row["episode_index"]): row["source_prefix"] for row in rows}
indices = list(range(start, end + 1))
if len(indices) != expected:
    raise RuntimeError(f"episode count mismatch: {len(indices)} != {expected}")
observed_tasks = {by_index[index].split("/")[-2] for index in indices}
if observed_tasks != {task}:
    raise RuntimeError(f"task-range mismatch: {observed_tasks} != {{{task!r}}}")

policy_src = os.path.expanduser(
    "~/vla_bspline/lerobot/src/lerobot/policies/smolvla_spline/modeling_smolvla_spline.py"
)
stats_src = os.path.expanduser(
    "~/vla_bspline/lerobot/src/lerobot/policies/smolvla_spline/spline_stats_robocasa_comp_v2_n8h24.json"
)
manifest = {
    "schema": "robocasa_task_focus_train_v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "mode": mode,
    "task": task,
    "episode_range_inclusive": [start, end],
    "episode_count": expected,
    "training_episode_count": len(training_episodes),
    "training_episode_indices_sha256": hashlib.sha256(
        json.dumps(training_episodes, separators=(",", ":")).encode()
    ).hexdigest(),
    "episode_selection": (
        "pure Kettle clause labels {384,385}" if mode == "gran_pure" else "full task range"
    ),
    "seed": int(seed),
    "steps": int(steps),
    "base_step": int(base_step),
    "base_checkpoint": os.path.realpath(checkpoint),
    "base_model_sha256": sha256(os.path.join(checkpoint, "model.safetensors")),
    "base_config_sha256": sha256(os.path.join(checkpoint, "config.json")),
    "base_training_state_sha256": sha256(training_state_path),
    "dataset": os.path.realpath(dataset),
    "dataset_episode_metadata_sha256": sha256(episode_meta),
    "dataset_tasks_metadata_sha256": sha256(task_meta),
    "dataset_stats_sha256": sha256(dataset_stats),
    "source_policy_sha256": sha256(policy_src),
    "source_stats_sha256": sha256(stats_src),
    "optimizer_semantics": "fresh optimizer and fresh cosine schedule from immutable model weights",
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip(),
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
  --policy.spline_stats_file_v2=spline_stats_robocasa_comp_v2_n8h24.json \
  --policy.predict_duration=true \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --policy.pretrained_path="$BASE_CKPT" \
  --dataset.repo_id=ember-lab-berkeley/robocasa365-target-composite-seen \
  --dataset.root="$DATASET" \
  --dataset.episodes="$EPISODES" \
  --output_dir="$OUT" \
  --steps="$STEPS" \
  --batch_size=32 \
  --num_workers=16 \
  --seed="$SEED" \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false

test -f "$OUT/checkpoints/last/pretrained_model/model.safetensors"
cp "$MANIFEST" "$OUT/launch_manifest.json"
echo "completed task-focus mode=$MODE task=$TASK output=$OUT"
