#!/bin/bash
#SBATCH --job-name=lang_resume
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Fail-safe only: resume one timed-out cell of train_language_factorial.sbatch
# from its most recent LeRobot checkpoint. Do not submit while the original job
# is healthy. Usage: sbatch resume_language_factorial.sbatch A 1001

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

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

DATASET="$HOME/scratch/vla_bspline/libero_l10gran"
OUT="$HOME/scratch/vla_bspline/outputs/${ARM}langMixL10_clean_s${SEED}"
FINAL="$OUT/checkpoints/last/pretrained_model"
TRAIN_CONFIG="$FINAL/train_config.json"
SOURCE_ROOT="$HOME/vla_bspline/lerobot"
STATS="$SOURCE_ROOT/src/lerobot/policies/smolvla_spline/spline_stats_libero_v2_h24_n8.json"

test -s "$TRAIN_CONFIG"
test -d "$DATASET/meta"
ARM="$ARM" SEED="$SEED" TRAIN_CONFIG="$TRAIN_CONFIG" python - <<'PY'
import json
import os

cfg = json.load(open(os.environ["TRAIN_CONFIG"]))
assert int(cfg["seed"]) == int(os.environ["SEED"]), (cfg["seed"], os.environ["SEED"])
expected = "smolvla" if os.environ["ARM"] == "A" else "smolvla_spline"
policy = cfg.get("policy", {})
actual = policy.get("type") if isinstance(policy, dict) else None
if actual is not None:
    assert actual == expected, (actual, expected)
print("validated resume config:", expected, "seed", cfg["seed"])
PY

echo "=== resume job=${SLURM_JOB_ID} arm=${ARM} seed=${SEED} node=$(hostname) time=$(date -Is) ==="
python -m lerobot.scripts.lerobot_train \
  --config_path="$TRAIN_CONFIG" \
  --resume=true

test -s "$FINAL/model.safetensors"
REPRO="$OUT/reproducibility"
mkdir -p "$REPRO"
test -s "$REPRO/dataset_info.json" || cp "$DATASET/meta/info.json" "$REPRO/dataset_info.json"
test -s "$REPRO/tasks.parquet" || cp "$DATASET/meta/tasks.parquet" "$REPRO/tasks.parquet"
if [ "$ARM" = C ]; then
  test -s "$REPRO/spline_stats_libero_v2_h24_n8.json" || \
    cp "$STATS" "$REPRO/spline_stats_libero_v2_h24_n8.json"
fi

ARM="$ARM" SEED="$SEED" FINAL="$FINAL" OUT="$OUT" SOURCE_ROOT="$SOURCE_ROOT" python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

final = Path(os.environ["FINAL"])
out = Path(os.environ["OUT"])
cfg = json.load(open(final / "config.json"))
train = json.load(open(final / "train_config.json"))
expected = "smolvla" if os.environ["ARM"] == "A" else "smolvla_spline"
assert cfg["type"] == expected, (cfg["type"], expected)
assert int(train["seed"]) == int(os.environ["SEED"]), (train["seed"], os.environ["SEED"])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = {
    "arm": os.environ["ARM"],
    "seed": int(os.environ["SEED"]),
    "dataset_root": str(Path(os.environ["OUT"]).parent.parent / "libero_l10gran"),
    "dataset_info_sha256": sha256(out / "reproducibility/dataset_info.json"),
    "dataset_tasks_sha256": sha256(out / "reproducibility/tasks.parquet"),
    "policy_type": cfg["type"],
    "n_action_steps": cfg["n_action_steps"],
    "lerobot_commit": subprocess.check_output(
        ["git", "-C", os.environ["SOURCE_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "resume_slurm_job_id": os.environ["SLURM_JOB_ID"],
}
if os.environ["ARM"] == "C":
    frozen = out / "reproducibility/spline_stats_libero_v2_h24_n8.json"
    manifest.update(
        n_ctrl=cfg["n_ctrl"],
        min_seg=cfg["min_seg"],
        horizon_max=cfg["horizon_max"],
        spline_stats_sha256=sha256(frozen),
    )
manifest_path = out / "reproducibility/manifest.json"
if manifest_path.exists():
    existing = json.load(open(manifest_path))
    for key in ("arm", "seed", "dataset_info_sha256", "dataset_tasks_sha256", "policy_type"):
        assert existing[key] == manifest[key], (key, existing[key], manifest[key])
else:
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print("validated resumed final checkpoint and provenance")
PY
echo "=== completed resume job=${SLURM_JOB_ID} time=$(date -Is) ==="
