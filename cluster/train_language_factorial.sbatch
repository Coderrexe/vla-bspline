#!/bin/bash
#SBATCH --job-name=lang_factorial
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Train one cell of the causal language-head factorial.
#
# Usage:
#   sbatch train_language_factorial.sbatch A 1000
#   sbatch train_language_factorial.sbatch C 1001
#
# A = stock SmolVLA waypoint head, evaluated at 10 actions/replan.
# C = event-segmented n=8 B-spline head, evaluated at 5 actions/replan.
# Both see the exact same fixed libero_l10gran dataset.

set -euo pipefail

ARM=${1:?expected arm A or C}
SEED=${2:?expected integer training seed}
case "$ARM" in
  A|C) ;;
  *) echo "invalid arm: $ARM" >&2; exit 2 ;;
esac
case "$SEED" in
  ''|*[!0-9]*) echo "invalid seed: $SEED" >&2; exit 2 ;;
esac

echo "=== start job=${SLURM_JOB_ID} arm=${ARM} seed=${SEED} node=$(hostname) time=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable", file=sys.stderr)
    raise SystemExit(97)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.get_device_name(0))
PY

DATASET="$HOME/scratch/vla_bspline/libero_l10gran"
OUT="$HOME/scratch/vla_bspline/outputs/${ARM}langMixL10_clean_s${SEED}"
SOURCE_ROOT="$HOME/vla_bspline/lerobot"
STATS="$SOURCE_ROOT/src/lerobot/policies/smolvla_spline/spline_stats_libero_v2_h24_n8.json"

test -d "$DATASET/meta"
if [ -e "$OUT" ]; then
  echo "FATAL: refusing to overwrite existing output: $OUT" >&2
  exit 42
fi

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
    --policy.type=smolvla \
    --policy.n_action_steps=10 \
    "${COMMON_ARGS[@]}"
else
  test -f "$STATS"
  python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla_spline \
    --policy.predict_duration=true \
    --policy.n_ctrl=8 \
    --policy.min_seg=8 \
    --policy.horizon_max=24 \
    --policy.n_action_steps=5 \
    --policy.spline_stats_file_v2=spline_stats_libero_v2_h24_n8.json \
    "${COMMON_ARGS[@]}"
fi

FINAL="$OUT/checkpoints/last/pretrained_model"
test -s "$FINAL/model.safetensors"
mkdir -p "$OUT/reproducibility"
cp "$DATASET/meta/info.json" "$OUT/reproducibility/dataset_info.json"
cp "$DATASET/meta/tasks.parquet" "$OUT/reproducibility/tasks.parquet"
if [ "$ARM" = C ]; then
  cp "$STATS" "$OUT/reproducibility/spline_stats_libero_v2_h24_n8.json"
fi

ARM="$ARM" SEED="$SEED" DATASET="$DATASET" OUT="$OUT" SOURCE_ROOT="$SOURCE_ROOT" STATS="$STATS" \
python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

arm = os.environ["ARM"]
seed = int(os.environ["SEED"])
out = Path(os.environ["OUT"])
final = out / "checkpoints/last/pretrained_model"
cfg = json.loads((final / "config.json").read_text())
train_cfg = json.loads((final / "train_config.json").read_text())

assert train_cfg["seed"] == seed, (train_cfg["seed"], seed)
assert cfg["type"] == ("smolvla" if arm == "A" else "smolvla_spline")
assert cfg["n_action_steps"] == (10 if arm == "A" else 5)
if arm == "C":
    assert cfg["n_ctrl"] == 8
    assert cfg["min_seg"] == 8
    assert cfg["horizon_max"] == 24

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

source_root = Path(os.environ["SOURCE_ROOT"])
manifest = {
    "arm": arm,
    "seed": seed,
    "dataset_root": str(Path(os.environ["DATASET"]).resolve()),
    "dataset_info_sha256": sha256(out / "reproducibility/dataset_info.json"),
    "dataset_tasks_sha256": sha256(out / "reproducibility/tasks.parquet"),
    "policy_type": cfg["type"],
    "n_action_steps": cfg["n_action_steps"],
    "lerobot_commit": subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
}
if arm == "C":
    frozen = out / "reproducibility/spline_stats_libero_v2_h24_n8.json"
    manifest.update(
        n_ctrl=cfg["n_ctrl"],
        min_seg=cfg["min_seg"],
        horizon_max=cfg["horizon_max"],
        spline_stats_sha256=sha256(frozen),
    )
(out / "reproducibility/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

echo "=== completed job=${SLURM_JOB_ID} arm=${ARM} seed=${SEED} time=$(date -Is) ==="
