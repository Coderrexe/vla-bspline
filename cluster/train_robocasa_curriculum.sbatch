#!/bin/bash
# Focused RoboCasa365 recovery curriculum. This intentionally loads the existing
# warm task-level checkpoint without changing the installed policy code or its
# normalization contract.
#
# Usage: sbatch train_robocasa_curriculum.sbatch easy4[_gran] [seed] [steps]
#   easy4      = compound/task-level language
#   easy4_gran = existing sentence-level language overlay
# Both select KettleBoiling, RinseSinkBasin, ScrubCuttingBoard, and
# StackBowlsCabinet (2,029 episodes; four exact source-task ranges).
#
# This launcher never deletes an output directory and records the exact input
# artifacts and runtime before training.
#SBATCH --job-name=rc_curr4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

CURRICULUM=${1:-easy4}
SEED=${2:-1000}
STEPS=${3:-100000}
case "$CURRICULUM" in
  easy4)
    DATASET_NAME=robocasa_composite_seen
    WARM_NAME=rcwarmC_100k
    VARIANT=task
    ;;
  easy4_gran)
    DATASET_NAME=robocasa_composite_seen_gran
    WARM_NAME=rcwarmgranC_100k
    VARIANT=gran
    ;;
  *)
    echo "unsupported curriculum: $CURRICULUM" >&2
    exit 64
    ;;
esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

DATASET="$HOME/scratch/vla_bspline/$DATASET_NAME"
WARM_CKPT="$HOME/scratch/vla_bspline/outputs/$WARM_NAME/checkpoints/last/pretrained_model"
OUT="$HOME/scratch/vla_bspline/outputs/rccurr_easy4_${VARIANT}_C_s${SEED}_${STEPS}_${SLURM_JOB_ID}"
MANIFEST_DIR="$HOME/scratch/vla_bspline/manifests/robocasa_curriculum"
LAUNCH_MANIFEST="$MANIFEST_DIR/rccurr_easy4_${VARIANT}_C_s${SEED}_${STEPS}_${SLURM_JOB_ID}.json"

[[ -d "$DATASET" ]] || { echo "missing dataset: $DATASET" >&2; exit 66; }
[[ -f "$WARM_CKPT/model.safetensors" ]] || {
  echo "missing warm checkpoint: $WARM_CKPT" >&2
  exit 66
}
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite: $OUT" >&2; exit 73; }
mkdir -p "$MANIFEST_DIR"

# Exact source episode ranges, established by scanning task_index in the source
# Parquet files. They contain 501 + 509 + 504 + 515 episodes.
EPISODES=$(python - <<'PY'
import json
episodes = (
    list(range(1010, 1511))
    + list(range(3528, 4541))
    + list(range(5546, 6061))
)
assert len(episodes) == 2029 and len(set(episodes)) == 2029
print(json.dumps(episodes, separators=(",", ":")))
PY
)

python - "$LAUNCH_MANIFEST" "$DATASET" "$WARM_CKPT" "$CURRICULUM" "$SEED" "$STEPS" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys

import pyarrow.parquet as pq

manifest_path, dataset, ckpt, curriculum, seed, steps = sys.argv[1:]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

policy_src = os.path.expanduser(
    "~/vla_bspline/lerobot/src/lerobot/policies/"
    "smolvla_spline/modeling_smolvla_spline.py"
)
stats_src = os.path.expanduser(
    "~/vla_bspline/lerobot/src/lerobot/policies/smolvla_spline/"
    "spline_stats_robocasa_comp_v2_n8h24.json"
)
episode_meta = os.path.join(dataset, "meta/episodes/chunk-000/file-000.parquet")
tasks_meta = os.path.join(dataset, "meta/tasks.parquet")
dataset_stats = os.path.join(dataset, "meta/stats.json")
task_ranges = [
    ("KettleBoiling", 1010, 1510, 501),
    ("RinseSinkBasin", 3528, 4036, 509),
    ("ScrubCuttingBoard", 4037, 4540, 504),
    ("StackBowlsCabinet", 5546, 6060, 515),
]

# Refuse to train if a dataset revision no longer matches the pre-registered
# episode selection.  The granular overlay changes language only; source_prefix
# remains the immutable source-task identity used here.
episode_rows = pq.read_table(
    episode_meta, columns=["episode_index", "source_prefix"]
).to_pylist()
by_index = {int(row["episode_index"]): row["source_prefix"] for row in episode_rows}
for task, start, end, expected_count in task_ranges:
    indices = list(range(start, end + 1))
    assert len(indices) == expected_count
    observed = {by_index[index].split("/")[-2] for index in indices}
    if observed != {task}:
        raise RuntimeError(
            f"dataset task-range mismatch for {task}: {start}-{end} -> {observed}"
        )

manifest = {
    "schema": "robocasa_curriculum_train_v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "curriculum": curriculum,
    "seed": int(seed),
    "steps": int(steps),
    "dataset": os.path.realpath(dataset),
    "warm_checkpoint": os.path.realpath(ckpt),
    "warm_model_sha256": sha256(os.path.join(ckpt, "model.safetensors")),
    "warm_config_sha256": sha256(os.path.join(ckpt, "config.json")),
    "source_policy_sha256": sha256(policy_src),
    "source_stats_sha256": sha256(stats_src),
    "dataset_episode_metadata_sha256": sha256(episode_meta),
    "dataset_tasks_metadata_sha256": sha256(tasks_meta),
    "dataset_stats_sha256": sha256(dataset_stats),
    "task_ranges": [
        {"task": task, "start": start, "end": end, "count": count}
        for task, start, end, count in task_ranges
    ],
    "episode_count": 2029,
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip(),
}
with open(manifest_path, "x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
PY

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(97)
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
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
  --policy.pretrained_path="$WARM_CKPT" \
  --dataset.repo_id=ember-lab-berkeley/robocasa365-target-composite-seen \
  --dataset.root="$DATASET" \
  --dataset.episodes="$EPISODES" \
  --output_dir="$OUT" \
  --steps="$STEPS" \
  --batch_size=32 \
  --num_workers=24 \
  --seed="$SEED" \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false

test -f "$OUT/checkpoints/last/pretrained_model/model.safetensors"
cp "$LAUNCH_MANIFEST" "$OUT/launch_manifest.json"
echo "completed curriculum=$CURRICULUM seed=$SEED steps=$STEPS output=$OUT"
