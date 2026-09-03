#!/bin/bash
# Resume a corrected-contract curriculum after a planned backfill timeout.
# Usage: sbatch resume_robocasa_corrected_curriculum.sbatch OUTPUT_DIR
#SBATCH --job-name=rc_corr_resume
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail
OUT=${1:?exact corrected curriculum output required}
case "$(basename "$OUT")" in
  rccorrected_easy4_task_C_s*_*) ;;
  rccorrected_easy4_gran_C_s*_*) ;;
  *) echo "refusing unexpected output path: $OUT" >&2; exit 64 ;;
esac

SNAP="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
export PYTHONPATH="$SNAP/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="$OUT/checkpoints/last/pretrained_model/train_config.json"
POLICY_CONFIG="$OUT/checkpoints/last/pretrained_model/config.json"
STATE="$OUT/checkpoints/last/training_state/training_step.json"
MODEL="$OUT/checkpoints/last/pretrained_model/model.safetensors"
for FILE in "$CONFIG" "$POLICY_CONFIG" "$STATE" "$MODEL"; do
  [[ -s "$FILE" ]] || { echo "missing resume input: $FILE" >&2; exit 66; }
done

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

MANIFEST_DIR="$HOME/scratch/vla_bspline/manifests/robocasa_corrected_resume"
MANIFEST="$MANIFEST_DIR/$(basename "$OUT")_resume_${SLURM_JOB_ID}.json"
[[ ! -e "$MANIFEST" ]] || { echo "refusing manifest overwrite: $MANIFEST" >&2; exit 73; }
mkdir -p "$MANIFEST_DIR"

python - "$OUT" "$SNAP" "$MANIFEST" <<'PY'
import hashlib
import inspect
import json
import os
import sys

from lerobot.policies.smolvla_spline import event_targets, modeling_smolvla_spline

output, snapshot, manifest_path = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

last = os.path.realpath(os.path.join(output, "checkpoints/last"))
config_path = os.path.join(last, "pretrained_model/config.json")
state_path = os.path.join(last, "training_state/training_step.json")
model_path = os.path.join(last, "pretrained_model/model.safetensors")
config = json.load(open(config_path, encoding="utf-8"))
state = json.load(open(state_path, encoding="utf-8"))
embedded = config.get("embedded_spline_stats")
if not embedded:
    raise RuntimeError("refusing to resume a checkpoint without embedded corrected stats")
if embedded.get("boundary_semantics") != "production_event_index_k_exclusive":
    raise RuntimeError("resume checkpoint has wrong event boundary semantics")
if embedded.get("action_layout") != "robocasa12":
    raise RuntimeError("resume checkpoint has wrong action layout")
if sha256(inspect.getfile(event_targets)) != \
        "9335cae53e277aa76975d59f43c266b0a1beee17d12d7737fd2d09db3016e9f6":
    raise RuntimeError("resume imported the wrong event-target implementation")
expected_root = os.path.realpath(os.path.join(snapshot, "src"))
if not os.path.realpath(inspect.getfile(modeling_smolvla_spline)).startswith(expected_root + os.sep):
    raise RuntimeError("resume did not import the isolated policy snapshot")
manifest = {
    "schema": "robocasa_corrected_curriculum_resume_v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "output": os.path.realpath(output),
    "resume_checkpoint": last,
    "resume_step": int(state.get("step", state.get("training_step"))),
    "resume_model_sha256": sha256(model_path),
    "resume_config_sha256": sha256(config_path),
    "event_targets_sha256": sha256(inspect.getfile(event_targets)),
    "modeling_sha256": sha256(inspect.getfile(modeling_smolvla_spline)),
    "source_snapshot": os.path.realpath(snapshot),
}
with open(manifest_path, "x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(manifest, sort_keys=True))
PY

python -m lerobot.scripts.lerobot_train --config_path="$CONFIG" --resume=true
test -s "$OUT/checkpoints/last/pretrained_model/model.safetensors"
echo "completed corrected resume output=$OUT"
