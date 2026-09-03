#!/bin/bash
#SBATCH --job-name=libero_det_v2
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# LOCAL PROPOSAL ONLY (2026-08-21): do not submit until the v2 evaluator and
# this launcher are reviewed and synced together. It does not call or modify the
# deployed v1 evaluator. Run the same checkpoint/tag twice and compare every
# episode's executed_action_trace_sha256, success, steps, and policy_calls.
#
# Both runs execute sequentially on the same allocated GPU UUID.
# Usage: sbatch smoke_libero_locked_eval_v2.sbatch CKPT TAG

set -euo pipefail

CKPT=${1:?expected checkpoint pretrained_model directory}
TAG=${2:?expected filesystem-safe experiment tag}
TASK_ID=${TASK_ID:-2}
case "$TAG" in
  *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;;
esac
case "$TASK_ID" in
  ''|*[!0-9]*) echo "invalid task id: $TASK_ID" >&2; exit 2 ;;
esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -s "$CKPT/model.safetensors"
test -s "$CKPT/config.json"
SOURCE="$HOME/vla_bspline/cluster"
STAGE="$HOME/scratch/vla_bspline/code_snapshots/libero_det_v2_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/determinism_smoke"
OUT_A="$OUTDIR/${TAG}_a_${SLURM_JOB_ID}.json"
OUT_B="$OUTDIR/${TAG}_b_${SLURM_JOB_ID}.json"
SUMMARY="$OUTDIR/${TAG}_identity_${SLURM_JOB_ID}.json"
test ! -e "$STAGE"
for path in "$OUT_A" "$OUT_B" "$SUMMARY"; do test ! -e "$path"; done
mkdir -p "$STAGE" "$OUTDIR"
cp "$SOURCE/libero_locked_eval_v2.py" "$STAGE/"
cp "$SOURCE/deterministic_eval.py" "$STAGE/"
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable", file=sys.stderr)
    raise SystemExit(97)
print("gpu:", torch.cuda.get_device_name(0))
PY

echo "=== deterministic v2 sequential smoke tag=${TAG} job=${SLURM_JOB_ID} node=$(hostname) ==="
for out in "$OUT_A" "$OUT_B"; do
  python "$STAGE/libero_locked_eval_v2.py" \
    --ckpt "$CKPT" \
    --suite libero_10 \
    --task_ids "$TASK_ID" \
    --state_start 0 \
    --states_per_task 1 \
    --seed_base 100000 \
    --control_freq 20 \
    --out "$out"
done

test -s "$OUT_A"
test -s "$OUT_B"
OUT_A="$OUT_A" OUT_B="$OUT_B" SUMMARY="$SUMMARY" TASK_ID="$TASK_ID" python - <<'PY'
import json
import os
from pathlib import Path

results = [json.load(open(os.environ[name])) for name in ("OUT_A", "OUT_B")]
keys = (
    "success",
    "steps",
    "policy_calls",
    "policy_chunk_generations",
    "executed_action_count",
    "executed_action_trace_sha256",
)
records = []
for result in results:
    assert result["schema_version"] == 2
    assert result["protocol"] == "libero_explicit_init_state_deterministic_v2"
    assert result["task_ids"] == [int(os.environ["TASK_ID"])]
    assert result["state_ids"] == [0]
    assert result["n_episodes"] == 1
    assert result["determinism"]["deterministic_algorithms"] is True
    assert result["determinism"]["deterministic_algorithms_warn_only"] is False
    assert result["determinism"]["cudnn_benchmark"] is False
    assert result["determinism"]["cudnn_deterministic"] is True
    assert result["determinism"]["cuda_matmul_allow_tf32"] is False
    assert result["determinism"]["cudnn_allow_tf32"] is False
    episode = result["tasks"][0]["episodes"][0]
    assert episode["executed_action_count"] == episode["steps"]
    assert len(episode["executed_action_trace_sha256"]) == 64
    records.append({key: episode[key] for key in keys})
exact = records[0] == records[1]
summary = {
    "schema_version": 1,
    "protocol": "single_gpu_sequential_deterministic_v2_identity",
    "exact": exact,
    "run_a": records[0],
    "run_b": records[1],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "determinism": results[0]["determinism"],
}
Path(os.environ["SUMMARY"]).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not exact:
    raise SystemExit("deterministic v2 identity gate failed")
PY

test -s "$SUMMARY"
echo "=== completed deterministic v2 smoke job=${SLURM_JOB_ID} time=$(date -Is) ==="
