#!/bin/bash
# Decode-cadence sweep with task progress diagnostics.  A checkpoint variant is
# created by symlinking immutable source weights and changing config.json only.
# Usage: sbatch eval_robocasa_progress_nas.sbatch CKPT TASK EPS HORIZON TAG NAS [SEED]
#SBATCH --job-name=rc_prog_nas
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

CKPT=${1:?checkpoint required}
TASK=${2:?task required}
EPISODES=${3:-5}
MAX_STEPS=${4:?max steps required}
TAG=${5:?tag required}
NAS=${6:?n_action_steps required}
SEED=${7:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 64 ;; esac
case "$NAS" in ''|*[!0-9]*) echo "invalid n_action_steps: $NAS" >&2; exit 64 ;; esac
(( NAS > 0 && NAS <= 24 )) || { echo "n_action_steps out of range: $NAS" >&2; exit 64; }

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_eval_20260821/robocasa_progress_eval.py"
VARIANT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_variants"
VARIANT="$VARIANT_ROOT/${TAG}_${TASK}_nas${NAS}_${SLURM_JOB_ID}"
OUT_DIR="$HOME/scratch/vla_bspline/outputs/robocasa_progress"
OUT="$OUT_DIR/${TAG}_${TASK}_nas${NAS}_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -f "$SCRIPT" ]] || { echo "missing frozen evaluator: $SCRIPT" >&2; exit 66; }
[[ -f "$CKPT/model.safetensors" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 66; }
[[ ! -e "$VARIANT" ]] || { echo "refusing to overwrite: $VARIANT" >&2; exit 73; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite: $OUT" >&2; exit 73; }
mkdir -p "$VARIANT_ROOT" "$OUT_DIR" "$VARIANT"

CKPT="$CKPT" VARIANT="$VARIANT" NAS="$NAS" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["CKPT"]).resolve()
variant = Path(os.environ["VARIANT"])
nas = int(os.environ["NAS"])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

for item in source.iterdir():
    if item.name in {"config.json", "cadence_variant_manifest.json"}:
        continue
    (variant / item.name).symlink_to(item.resolve())

config = json.loads((source / "config.json").read_text())
original_nas = int(config["n_action_steps"])
config["n_action_steps"] = nas
(variant / "config.json").write_text(json.dumps(config, indent=2) + "\n")
manifest = {
    "schema": "robocasa_cadence_variant_v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "source_checkpoint": str(source),
    "source_model_sha256": sha256(source / "model.safetensors"),
    "source_config_sha256": sha256(source / "config.json"),
    "original_n_action_steps": original_nas,
    "evaluated_n_action_steps": nas,
}
(variant / "cadence_variant_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

python "$SCRIPT" \
  --ckpt "$VARIANT" \
  --task "$TASK" \
  --episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" \
  --seed_base "$SEED" \
  --out "$OUT"

test -s "$OUT"
sha256sum "$SCRIPT" "$VARIANT/config.json" "$VARIANT/cadence_variant_manifest.json" "$OUT"
