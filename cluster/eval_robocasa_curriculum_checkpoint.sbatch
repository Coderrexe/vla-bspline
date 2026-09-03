#!/bin/bash
# Evaluate one immutable intermediate RoboCasa curriculum checkpoint with a
# crossed decode cadence and instruction scheduler.
# Usage:
#   sbatch eval_robocasa_curriculum_checkpoint.sbatch \
#     RUN_DIR STEP TASK EPISODES MAX_STEPS TAG NAS MODE [SEED]
# MODE: env, selfpaced, proprio, oracle, or fixedk:<K>.
#
# Submission must occur only after the parent process verifies that the source
# checkpoint is fully written (training_step matches and two spaced model hashes
# agree).  This job independently revalidates the step and records provenance.
#SBATCH --job-name=rc_curr_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

RUN_DIR=${1:?curriculum run directory required}
STEP=${2:?checkpoint step required}
TASK=${3:?task required}
EPISODES=${4:-5}
MAX_STEPS=${5:?max steps required}
TAG=${6:?tag required}
NAS=${7:?n_action_steps required}
MODE=${8:-env}
SEED=${9:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 64 ;; esac
case "$NAS" in ''|*[!0-9]*) echo "invalid n_action_steps: $NAS" >&2; exit 64 ;; esac
(( NAS > 0 && NAS <= 24 )) || { echo "n_action_steps out of range: $NAS" >&2; exit 64; }
printf -v STEP_DIR '%06d' "$STEP"

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

SOURCE="$RUN_DIR/checkpoints/$STEP_DIR/pretrained_model"
TRAIN_STATE="$RUN_DIR/checkpoints/$STEP_DIR/training_state/training_step.json"
SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_eval_v3_20260821/robocasa_progress_eval.py"
VARIANT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_curriculum_variants"
VARIANT="$VARIANT_ROOT/${TAG}_${TASK}_step${STEP_DIR}_nas${NAS}_${SLURM_JOB_ID}"
OUT_DIR="$HOME/scratch/vla_bspline/outputs/robocasa_curriculum_eval"
SAFE_MODE=${MODE//:/_}
OUT="$OUT_DIR/${TAG}_${TASK}_step${STEP_DIR}_nas${NAS}_${SAFE_MODE}_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -f "$SCRIPT" ]] || { echo "missing frozen evaluator: $SCRIPT" >&2; exit 66; }
[[ -f "$SOURCE/model.safetensors" ]] || { echo "missing source checkpoint: $SOURCE" >&2; exit 66; }
[[ -f "$TRAIN_STATE" ]] || { echo "missing training state: $TRAIN_STATE" >&2; exit 66; }
[[ ! -e "$VARIANT" ]] || { echo "refusing to overwrite: $VARIANT" >&2; exit 73; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite: $OUT" >&2; exit 73; }
mkdir -p "$VARIANT_ROOT" "$OUT_DIR" "$VARIANT"

SOURCE="$SOURCE" TRAIN_STATE="$TRAIN_STATE" VARIANT="$VARIANT" NAS="$NAS" EXPECTED_STEP="$STEP" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE"]).resolve()
training_state_path = Path(os.environ["TRAIN_STATE"]).resolve()
variant = Path(os.environ["VARIANT"])
nas = int(os.environ["NAS"])
expected_step = int(os.environ["EXPECTED_STEP"])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

training_state = json.loads(training_state_path.read_text())
observed_step = training_state.get("step", training_state.get("training_step"))
if int(observed_step) != expected_step:
    raise RuntimeError(
        f"checkpoint step mismatch: expected {expected_step}, observed {observed_step}"
    )

for item in source.iterdir():
    if item.name in {"config.json", "curriculum_eval_variant_manifest.json"}:
        continue
    (variant / item.name).symlink_to(item.resolve())

config = json.loads((source / "config.json").read_text())
original_nas = int(config["n_action_steps"])
config["n_action_steps"] = nas
(variant / "config.json").write_text(json.dumps(config, indent=2) + "\n")
manifest = {
    "schema": "robocasa_curriculum_eval_variant_v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "source_checkpoint": str(source),
    "source_model_sha256": sha256(source / "model.safetensors"),
    "source_config_sha256": sha256(source / "config.json"),
    "training_state_sha256": sha256(training_state_path),
    "checkpoint_step": expected_step,
    "original_n_action_steps": original_nas,
    "evaluated_n_action_steps": nas,
}
(variant / "curriculum_eval_variant_manifest.json").write_text(
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
  --instr_mode "$MODE" \
  --out "$OUT"

test -s "$OUT"
sha256sum \
  "$SCRIPT" \
  "$VARIANT/config.json" \
  "$VARIANT/curriculum_eval_variant_manifest.json" \
  "$OUT"
