#!/bin/bash
# Matched RoboCasa365 task/granular recovery under the corrected production
# event-target and normalization contract.  This launcher deliberately imports
# an immutable, isolated LeRobot source snapshot: it never edits or imports the
# mutable installed spline policy tree.
#
# Usage: sbatch train_robocasa_corrected_curriculum.sbatch task|gran [seed] [steps]
#
# Both arms start from the exact same atomic C checkpoint and consume identical
# physical episodes.  The sole dataset delta is the existing frame-level
# language overlay in the granular arm.
#SBATCH --job-name=rc_corr4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail

ARM=${1:?expected task or gran}
SEED=${2:-1000}
STEPS=${3:-100000}
case "$ARM" in
  task) DATASET_NAME=robocasa_composite_seen ;;
  gran) DATASET_NAME=robocasa_composite_seen_gran ;;
  *) echo "unsupported arm: $ARM" >&2; exit 64 ;;
esac
for VALUE in "$SEED" "$STEPS"; do
  case "$VALUE" in ''|*[!0-9]*) echo "invalid integer: $VALUE" >&2; exit 64 ;; esac
done

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

SNAP="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
export PYTHONPATH="$SNAP/src${PYTHONPATH:+:$PYTHONPATH}"
POLICY_DIR="$SNAP/src/lerobot/policies/smolvla_spline"
POLICY_SRC="$POLICY_DIR/modeling_smolvla_spline.py"
CONFIG_SRC="$POLICY_DIR/configuration_smolvla_spline.py"
EVENT_SRC="$POLICY_DIR/event_targets.py"
CONTRACT_SRC="$POLICY_DIR/stats_contract.py"
STATS_NAME=spline_stats_robocasa_composite_seen_all_v2_n8h24.json
STATS_SRC="$POLICY_DIR/$STATS_NAME"
DATASET="$HOME/scratch/vla_bspline/$DATASET_NAME"
INIT="$HOME/scratch/vla_bspline/outputs/rcC_100k/checkpoints/last/pretrained_model"
OUT="$HOME/scratch/vla_bspline/outputs/rccorrected_easy4_${ARM}_C_s${SEED}_${STEPS}_${SLURM_JOB_ID}"
MANIFEST_DIR="$HOME/scratch/vla_bspline/manifests/robocasa_corrected_curriculum"
MANIFEST="$MANIFEST_DIR/rccorrected_easy4_${ARM}_C_s${SEED}_${STEPS}_${SLURM_JOB_ID}.json"

for FILE in "$POLICY_SRC" "$CONFIG_SRC" "$EVENT_SRC" "$CONTRACT_SRC" \
  "$STATS_SRC" "$INIT/model.safetensors" "$INIT/config.json"; do
  [[ -f "$FILE" ]] || { echo "missing immutable input: $FILE" >&2; exit 66; }
done
[[ -d "$DATASET" ]] || { echo "missing dataset: $DATASET" >&2; exit 66; }
[[ ! -e "$OUT" ]] || { echo "refusing output overwrite: $OUT" >&2; exit 73; }
[[ ! -e "$MANIFEST" ]] || { echo "refusing manifest overwrite: $MANIFEST" >&2; exit 73; }
mkdir -p "$MANIFEST_DIR"

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

from lerobot.policies.smolvla_spline import event_targets, stats_contract
from lerobot.policies.smolvla_spline import modeling_smolvla_spline

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
expected_modeling = os.path.join(policy_dir, "modeling_smolvla_spline.py")
if os.path.realpath(inspect.getfile(modeling_smolvla_spline)) != os.path.realpath(expected_modeling):
    raise RuntimeError("isolated PYTHONPATH did not bind the frozen spline policy")
if sha256(os.path.join(policy_dir, stats_name)) != \
        "287be4152eb77bd9301a8d62e26123574356f9cb543e1b8050406e783ec78007":
    raise RuntimeError("corrected composite statistics hash mismatch")
if sha256(inspect.getfile(event_targets)) != \
        "9335cae53e277aa76975d59f43c266b0a1beee17d12d7737fd2d09db3016e9f6":
    raise RuntimeError("production event-target source hash mismatch")

episode_meta = os.path.join(dataset, "meta/episodes/chunk-000/file-000.parquet")
tasks_meta = os.path.join(dataset, "meta/tasks.parquet")
dataset_stats = os.path.join(dataset, "meta/stats.json")
ranges = [
    ("KettleBoiling", 1010, 1510, 501),
    ("RinseSinkBasin", 3528, 4036, 509),
    ("ScrubCuttingBoard", 4037, 4540, 504),
    ("StackBowlsCabinet", 5546, 6060, 515),
]
rows = pq.read_table(episode_meta, columns=["episode_index", "source_prefix"]).to_pylist()
by_index = {int(row["episode_index"]): row["source_prefix"] for row in rows}
for task, start, end, count in ranges:
    indices = list(range(start, end + 1))
    if len(indices) != count:
        raise RuntimeError(f"range count mismatch for {task}")
    observed = {by_index[index].split("/")[-2] for index in indices}
    if observed != {task}:
        raise RuntimeError(f"task provenance mismatch for {task}: {observed}")
if sorted(episodes) != sorted(
    index for _, start, end, _ in ranges for index in range(start, end + 1)
):
    raise RuntimeError("training episode list differs from audited ranges")

# Bind and hash the exact physical transition stream.  The pre-existing audit
# proves equality between arms; each launch records its own immutable digest.
physical_columns = [
    "index", "episode_index", "frame_index", "timestamp", "action",
    "observation.state", "next.reward", "next.done",
]
table = ds.dataset(os.path.join(dataset, "data"), format="parquet").to_table(
    columns=physical_columns,
    filter=(
        ((ds.field("episode_index") >= 1010) & (ds.field("episode_index") <= 1510))
        | ((ds.field("episode_index") >= 3528) & (ds.field("episode_index") <= 4540))
        | ((ds.field("episode_index") >= 5546) & (ds.field("episode_index") <= 6060))
    ),
)
transition_hash = hashlib.sha256()
for batch in table.to_batches(max_chunksize=32768):
    transition_hash.update(batch.serialize().to_pybytes())

source_files = {
    "modeling": expected_modeling,
    "configuration": os.path.join(policy_dir, "configuration_smolvla_spline.py"),
    "event_targets": inspect.getfile(event_targets),
    "stats_contract": inspect.getfile(stats_contract),
    "corrected_stats": os.path.join(policy_dir, stats_name),
}
manifest = {
    "schema": "robocasa_corrected_curriculum_train_v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "arm": arm,
    "seed": int(seed),
    "steps": int(steps),
    "initialization": os.path.realpath(init),
    "initial_model_sha256": sha256(os.path.join(init, "model.safetensors")),
    "initial_config_sha256": sha256(os.path.join(init, "config.json")),
    "dataset": os.path.realpath(dataset),
    "dataset_episode_metadata_sha256": sha256(episode_meta),
    "dataset_tasks_metadata_sha256": sha256(tasks_meta),
    "dataset_stats_sha256": sha256(dataset_stats),
    "selected_physical_transition_sha256": transition_hash.hexdigest(),
    "episode_count": len(episodes),
    "task_ranges": [
        {"task": task, "start": start, "end": end, "count": count}
        for task, start, end, count in ranges
    ],
    "source_snapshot": os.path.realpath(snapshot),
    "source_sha256": {name: sha256(path) for name, path in source_files.items()},
    "normalization_contract": {
        "stats_sha256": sha256(source_files["corrected_stats"]),
        "event_boundary_semantics": stats_contract.EVENT_BOUNDARY_SEMANTICS,
        "checkpoint_embedding_required": True,
    },
    "output": os.path.realpath(output),
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
  --num_workers=24 \
  --seed="$SEED" \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false

test -f "$OUT/checkpoints/last/pretrained_model/model.safetensors"
python - "$OUT" "$MANIFEST" <<'PY'
import json
import os
import shutil
import sys

output, manifest = sys.argv[1:]
config_path = os.path.join(output, "checkpoints/last/pretrained_model/config.json")
config = json.load(open(config_path, encoding="utf-8"))
embedded = config.get("embedded_spline_stats")
if not embedded:
    raise RuntimeError("corrected normalization was not embedded in saved checkpoint")
if embedded.get("boundary_semantics") != "production_event_index_k_exclusive":
    raise RuntimeError("saved checkpoint has wrong event-boundary contract")
if embedded.get("action_layout") != "robocasa12":
    raise RuntimeError("saved checkpoint has wrong action layout contract")
destination = os.path.join(output, "launch_manifest.json")
if os.path.exists(destination):
    raise FileExistsError(destination)
shutil.copyfile(manifest, destination)
PY

echo "completed corrected-contract arm=$ARM seed=$SEED output=$OUT"
