#!/bin/bash
#SBATCH --job-name=duration_cal_v2
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Reviewed proposal only: this file does not submit itself.
# Usage: sbatch cluster/calibrate_duration_head_v2.sbatch \
#   CKPT TAG DATASET_ROOT TASKS_SHA256
#
# Prediction calibration MUST use the checkpoint's original language metadata.
# The action-identical libero_l10gran overlay is valid for an action-only prior,
# but invalid here because the duration head consumes language.  The pinned
# official LIBERO tasks hash currently on Misha is:
# c01ec32de3ba5f5cd7eb251ce1e6279a60fbe1c80d12df1eeca746927f78df6c

set -euo pipefail

CKPT=${1:?expected spline checkpoint pretrained_model directory}
TAG=${2:?expected filesystem-safe tag}
DATASET_ROOT=${3:?expected exact original training dataset root}
TASKS_SHA256=${4:?expected sha256 of original meta/tasks.parquet}
case "$TAG" in
  *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;;
esac
if [ "${#TASKS_SHA256}" -ne 64 ]; then
  echo "TASKS_SHA256 must be 64 lowercase hex characters" >&2
  exit 2
fi
case "$TASKS_SHA256" in
  *[!0-9a-f]*) echo "TASKS_SHA256 must be 64 lowercase hex characters" >&2; exit 2 ;;
esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable", file=sys.stderr)
    raise SystemExit(97)
print("gpu:", torch.cuda.get_device_name(0))
PY

SOURCE="$HOME/vla_bspline"
STAGE="$HOME/scratch/vla_bspline/code_snapshots/duration_cal_v2_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/duration_calibration_v2"
MANIFEST="$OUTDIR/${TAG}_${SLURM_JOB_ID}_manifest.json"
PREDICTIONS="$OUTDIR/${TAG}_${SLURM_JOB_ID}_predictions.json"
CALIBRATOR="$OUTDIR/${TAG}_${SLURM_JOB_ID}_calibrator.json"
REPORT="$OUTDIR/${TAG}_${SLURM_JOB_ID}_report.json"
test -s "$CKPT/config.json"
test -s "$CKPT/train_config.json"
test -s "$DATASET_ROOT/meta/tasks.parquet"
test ! -e "$STAGE"
test ! -e "$MANIFEST"
test ! -e "$PREDICTIONS"
test ! -e "$CALIBRATOR"
test ! -e "$REPORT"
mkdir -p "$STAGE" "$OUTDIR"
cp "$SOURCE/scripts/analysis/duration_head_calibration_v2.py" "$STAGE/"
cp "$SOURCE/scripts/analysis/duration_calibration_core.py" "$STAGE/"
cp "$SOURCE/scripts/analysis/duration_training_prior.py" "$STAGE/"
cp "$SOURCE/policy/smolvla_spline/event_targets.py" "$STAGE/"
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"
chmod -R a-w "$STAGE"

echo "=== prepare exact action-only manifest ==="
python "$STAGE/duration_head_calibration_v2.py" prepare \
  --ckpt "$CKPT" \
  --dataset_root "$DATASET_ROOT" \
  --dataset_repo_id HuggingFaceVLA/libero \
  --expected_tasks_sha256 "$TASKS_SHA256" \
  --event_target_source "$STAGE/event_targets.py" \
  --samples_per_task_fold 32 \
  --out "$MANIFEST"

echo "=== selected-frame inference (only stage that decodes video) ==="
python "$STAGE/duration_head_calibration_v2.py" infer \
  --manifest "$MANIFEST" \
  --draws 4 \
  --batch_size 16 \
  --num_workers "$SLURM_CPUS_PER_TASK" \
  --seed 1701 \
  --out "$PREDICTIONS"

echo "=== locked fit/select/audit analysis ==="
python "$STAGE/duration_head_calibration_v2.py" analyze \
  --predictions "$PREDICTIONS" \
  --calibrator_out "$CALIBRATOR" \
  --out "$REPORT"

REPORT="$REPORT" CALIBRATOR="$CALIBRATOR" python - <<'PY'
import json
import os

report = json.load(open(os.environ["REPORT"]))
calibrator = json.load(open(os.environ["CALIBRATOR"]))
assert report["protocol"] == "libero_duration_head_calibration_train_only_v2"
assert report["stage"] == "calibration_analysis"
assert calibrator["excluded_fold"] == "audit"
assert calibrator["shape_channels_modified"] is False
assert calibrator["selected_family"] == report["selected_family"]
print(
    "validated:", report["selected_family"],
    report["audit_opened_once"]["identity"]["mae"], "->",
    report["audit_opened_once"]["selected"]["mae"],
)
PY

echo "=== completed duration calibration job=${SLURM_JOB_ID} time=$(date -Is) ==="

