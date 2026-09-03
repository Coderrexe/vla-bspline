#!/bin/bash
#SBATCH --job-name=train_lang_shuffle
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# PROPOSAL ONLY.  Cross this control with BOTH A and C and the same predeclared
# seeds; a C-only run cannot establish B-spline specificity.  Submit only after
# the builder job succeeds and its immutable manifest is reviewed.

set -euo pipefail

ARM=${1:?expected A or C}
SEED=${2:?expected integer seed}
case "$ARM" in A|C) ;; *) echo "invalid arm: $ARM" >&2; exit 2 ;; esac
case "$SEED" in ''|*[!0-9]*) echo "invalid seed: $SEED" >&2; exit 2 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

DATASET="$HOME/scratch/vla_bspline/libero_l10gran_semshuffle_s1701"
MANIFEST="$DATASET/_semantic_shuffle_control/manifest.json"
OUT="$HOME/scratch/vla_bspline/outputs/${ARM}langShuffleL10_s${SEED}"
SOURCE_ROOT="$HOME/vla_bspline/lerobot"
BUILDER="$HOME/vla_bspline/cluster/build_language_shuffle_control.py"
STATS="$SOURCE_ROOT/src/lerobot/policies/smolvla_spline/spline_stats_libero_v2_h24_n8.json"

test -s "$MANIFEST"
test -s "$BUILDER"
test ! -e "$OUT"
MANIFEST="$MANIFEST" DATASET="$DATASET" BUILDER="$BUILDER" python - <<'PY'
import hashlib, importlib.util, json, os
from pathlib import Path

p = json.load(open(os.environ["MANIFEST"]))
tasks = Path(os.environ["DATASET"]) / "meta/tasks.parquet"
h = hashlib.sha256(tasks.read_bytes()).hexdigest()
assert p["control_kind"] == "same-support-caption-derangement"
assert p["derived_tasks_sha256"] == h
assert p["normalized_text_collision_count"] == 0
assert p["deranged_fraction_of_appended_frames"] > 0.0

# Re-scan the zero-copy source before consuming a GPU. This catches a changed
# task table or task-index schedule behind the overlay symlinks.
builder_path = Path(os.environ["BUILDER"])
assert p["builder_sha256"] == hashlib.sha256(builder_path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location("semshuffle_verify", builder_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
current = module.analyze(
    Path(p["source_root"]), int(p["original_task_count"]), int(p["seed"])
)
assert current["source_tasks_sha256"] == p["source_tasks_sha256"]
assert current["task_index_schedule_sha256"] == p["task_index_schedule_sha256"]
PY

COMMON_ARGS=(
  --policy.push_to_hub=false
  --policy.load_vlm_weights=true
  --dataset.repo_id=HuggingFaceVLA/libero
  --dataset.root="$DATASET"
  --output_dir="$OUT"
  --steps=100000
  --batch_size=32
  --num_workers=24
  --seed="$SEED"
  --save_freq=50000
  --log_freq=100
  --wandb.enable=false
)

if [ "$ARM" = A ]; then
  python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla --policy.n_action_steps=10 "${COMMON_ARGS[@]}"
else
  test -s "$STATS"
  python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla_spline --policy.predict_duration=true \
    --policy.n_ctrl=8 --policy.min_seg=8 --policy.horizon_max=24 \
    --policy.n_action_steps=5 \
    --policy.spline_stats_file_v2=spline_stats_libero_v2_h24_n8.json \
    "${COMMON_ARGS[@]}"
fi

FINAL="$OUT/checkpoints/last/pretrained_model"
test -s "$FINAL/model.safetensors"
mkdir -p "$OUT/reproducibility"
cp "$MANIFEST" "$OUT/reproducibility/semantic_shuffle_manifest.json"
cp "$DATASET/meta/tasks.parquet" "$OUT/reproducibility/tasks.parquet"
ARM="$ARM" SEED="$SEED" FINAL="$FINAL" python - <<'PY'
import json, os
from pathlib import Path

arm = os.environ["ARM"]
seed = int(os.environ["SEED"])
final = Path(os.environ["FINAL"])
cfg = json.loads((final / "config.json").read_text())
train_cfg = json.loads((final / "train_config.json").read_text())
assert train_cfg["seed"] == seed
assert cfg["type"] == ("smolvla" if arm == "A" else "smolvla_spline")
assert cfg["n_action_steps"] == (10 if arm == "A" else 5)
if arm == "C":
    assert (cfg["n_ctrl"], cfg["min_seg"], cfg["horizon_max"]) == (8, 8, 24)
PY
echo "=== completed shuffle training job=${SLURM_JOB_ID} arm=${ARM} seed=${SEED} ==="
