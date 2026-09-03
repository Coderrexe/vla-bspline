#!/bin/bash
# Read-only decode-rate intervention on an immutable RoboCasa curriculum checkpoint.
# Usage: sbatch ... RUN STEP TASK EPISODES MAX_STEPS TAG NAS MODE RATE [SEED]
# RATE is exec_rate_ratio: >1 executes the same predicted spline more slowly.
#SBATCH --job-name=rc_rate_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

RUN_DIR=${1:?run required}
STEP=${2:?step required}
TASK=${3:?task required}
EPISODES=${4:-5}
MAX_STEPS=${5:?max steps required}
TAG=${6:?tag required}
NAS=${7:?n_action_steps required}
MODE=${8:?instruction mode required}
RATE=${9:?exec_rate_ratio required}
SEED=${10:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 64 ;; esac
case "$NAS" in ''|*[!0-9]*) echo "invalid n_action_steps: $NAS" >&2; exit 64 ;; esac
printf -v STEP_DIR '%06d' "$STEP"

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

SOURCE="$RUN_DIR/checkpoints/$STEP_DIR/pretrained_model"
TRAIN_STATE="$RUN_DIR/checkpoints/$STEP_DIR/training_state/training_step.json"
SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_eval_v4_20260821/robocasa_progress_eval.py"
VARIANT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_curriculum_variants"
SAFE_RATE=${RATE//./p}
SAFE_MODE=${MODE//:/_}
VARIANT="$VARIANT_ROOT/${TAG}_${TASK}_step${STEP_DIR}_nas${NAS}_rate${SAFE_RATE}_${SLURM_JOB_ID}"
OUT_DIR="$HOME/scratch/vla_bspline/outputs/robocasa_curriculum_rate_eval"
OUT="$OUT_DIR/${TAG}_${TASK}_step${STEP_DIR}_nas${NAS}_rate${SAFE_RATE}_${SAFE_MODE}_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -f "$SCRIPT" ]] || { echo "missing frozen evaluator" >&2; exit 66; }
[[ -f "$SOURCE/model.safetensors" ]] || { echo "missing source checkpoint" >&2; exit 66; }
[[ -f "$TRAIN_STATE" ]] || { echo "missing training state" >&2; exit 66; }
[[ ! -e "$VARIANT" ]] || { echo "refusing variant overwrite" >&2; exit 73; }
[[ ! -e "$OUT" ]] || { echo "refusing output overwrite" >&2; exit 73; }
mkdir -p "$VARIANT_ROOT" "$OUT_DIR" "$VARIANT"

SOURCE="$SOURCE" TRAIN_STATE="$TRAIN_STATE" VARIANT="$VARIANT" NAS="$NAS" RATE="$RATE" EXPECTED_STEP="$STEP" python - <<'PY'
import hashlib, json, os
from pathlib import Path

source = Path(os.environ["SOURCE"]).resolve()
state_path = Path(os.environ["TRAIN_STATE"]).resolve()
variant = Path(os.environ["VARIANT"])
nas = int(os.environ["NAS"])
rate = float(os.environ["RATE"])
expected = int(os.environ["EXPECTED_STEP"])
if not (0.25 <= rate <= 4.0):
    raise ValueError(f"exec_rate_ratio out of safe range: {rate}")
if not (1 <= nas <= 24):
    raise ValueError(f"n_action_steps out of range: {nas}")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

state = json.loads(state_path.read_text())
observed = int(state.get("step", state.get("training_step")))
if observed != expected:
    raise RuntimeError(f"step mismatch: {observed} != {expected}")
for item in source.iterdir():
    if item.name != "config.json":
        (variant / item.name).symlink_to(item.resolve())
config = json.loads((source / "config.json").read_text())
original_nas = int(config["n_action_steps"])
original_rate = float(config.get("exec_rate_ratio", 1.0))
config["n_action_steps"] = nas
config["exec_rate_ratio"] = rate
(variant / "config.json").write_text(json.dumps(config, indent=2) + "\n")
manifest = {
    "schema": "robocasa_curriculum_rate_variant_v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "source_checkpoint": str(source),
    "checkpoint_step": expected,
    "source_model_sha256": sha256(source / "model.safetensors"),
    "source_config_sha256": sha256(source / "config.json"),
    "training_state_sha256": sha256(state_path),
    "original_n_action_steps": original_nas,
    "evaluated_n_action_steps": nas,
    "original_exec_rate_ratio": original_rate,
    "evaluated_exec_rate_ratio": rate,
}
(variant / "curriculum_rate_variant_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

python "$SCRIPT" \
  --ckpt "$VARIANT" --task "$TASK" --episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" --seed_base "$SEED" --instr_mode "$MODE" --out "$OUT"

test -s "$OUT"
sha256sum "$SCRIPT" "$VARIANT/config.json" "$VARIANT/curriculum_rate_variant_manifest.json" "$OUT"
