#!/bin/bash
#SBATCH --job-name=dur_ablate_v2
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=16:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Reviewed for the 2026-08-21 same-process determinism smoke. Claim-scale runs
# remain gated on exact replay of the targeted task/state smoke.
#
# One process loads the policy/environments once, executes two predicted passes
# as an exact gate, and only then executes fixed and shuffled. Separate jobs or
# processes are intentionally unsupported because deterministic-v2 replay was
# not exact across process reloads on the same GPU.
#
# Usage:
#   sbatch eval_duration_ablation_v2.sbatch CKPT TAG FIXED_MANIFEST.json
# Optional environment:
#   TASK_IDS=0,4 STATES_PER_TASK=50 STATE_START=0 SEED_BASE=100000

set -euo pipefail

CKPT=${1:?expected duration-enabled checkpoint pretrained_model directory}
TAG=${2:?expected filesystem-safe experiment tag}
FIXED_MANIFEST=${3:?expected complete fixed-duration training-prior manifest}
TASK_IDS=${TASK_IDS:-}
STATE_START=${STATE_START:-0}
STATES_PER_TASK=${STATES_PER_TASK:-50}
SEED_BASE=${SEED_BASE:-100000}
case "$TAG" in
  *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;;
esac
for value in "$STATE_START" "$STATES_PER_TASK" "$SEED_BASE"; do
  case "$value" in
    ''|*[!0-9]*) echo "invalid nonnegative integer: $value" >&2; exit 2 ;;
  esac
done
if [ "$STATES_PER_TASK" -le 0 ]; then
  echo "STATES_PER_TASK must be positive" >&2
  exit 2
fi
if [ -n "$TASK_IDS" ]; then
  case "$TASK_IDS" in
    *[!0-9,]*) echo "TASK_IDS must be comma-separated nonnegative integers" >&2; exit 2 ;;
  esac
fi

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
test -s "$FIXED_MANIFEST"
SOURCE="$HOME/vla_bspline/cluster"
STAGE="$HOME/scratch/vla_bspline/code_snapshots/duration_v2_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/duration_ablation_v2"
OUT="$OUTDIR/${TAG}_${SLURM_JOB_ID}.json"
test ! -e "$STAGE"
test ! -e "$OUT"
mkdir -p "$STAGE" "$OUTDIR"
for name in \
  libero_duration_ablation_v2.py \
  duration_intervention_v2.py \
  libero_locked_eval_v2.py \
  deterministic_eval.py
do
  cp "$SOURCE/$name" "$STAGE/$name"
done
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable", file=sys.stderr)
    raise SystemExit(97)
print("gpu:", torch.cuda.get_device_name(0))
PY

ARGS=(
  --ckpt "$CKPT"
  --fixed_manifest "$FIXED_MANIFEST"
  --suite libero_10
  --state_start "$STATE_START"
  --states_per_task "$STATES_PER_TASK"
  --seed_base "$SEED_BASE"
  --control_freq 20
  --shuffle_seed 73921
  --out "$OUT"
)
if [ -n "$TASK_IDS" ]; then
  ARGS+=(--task_ids "$TASK_IDS")
fi

echo "=== duration-v2 tag=${TAG} job=${SLURM_JOB_ID} node=$(hostname) ==="
set +e
python "$STAGE/libero_duration_ablation_v2.py" "${ARGS[@]}"
EVAL_STATUS=$?
set -e
test -s "$OUT"

OUT="$OUT" STATES_PER_TASK="$STATES_PER_TASK" python - <<'PY'
import json
import os

result = json.load(open(os.environ["OUT"]))
assert result["schema_version"] == 2
assert result["protocol"] == "libero_duration_intervention_deterministic_v2"
assert result["duration_modes"] == ["predicted", "fixed", "shuffled"]
assert result["fixed_manifest_complete_training_scan"] is True
expected = len(result["task_ids"]) * int(os.environ["STATES_PER_TASK"])
for name in ("predicted", "predicted_repeat"):
    assert result["arms"][name]["n_episodes"] == expected
if result["status"] == "complete":
    assert result["determinism_gate"]["exact"] is True
    assert all(
        gate["exact"]
        for gate in result["intervention_initial_state_gates"].values()
    )
    assert set(result["arms"]) == {
        "predicted", "predicted_repeat", "fixed", "shuffled"
    }
    for name in ("fixed", "shuffled"):
        assert result["arms"][name]["n_episodes"] == expected
    assert result["shuffle_source_manifest"]["manifest_sha256"]
else:
    assert result["status"] in {
        "determinism_gate_failed", "intervention_initial_state_gate_failed"
    }
    if result["status"] == "determinism_gate_failed":
        assert result["determinism_gate"]["exact"] is False
        assert set(result["arms"]) == {"predicted", "predicted_repeat"}
print(
    "validated:", result["status"],
    "gate divergences=", result["determinism_gate"]["n_divergent_episodes"],
)
PY

if [ "$EVAL_STATUS" -ne 0 ]; then
  echo "duration-v2 evaluator failed after publishing diagnostic artifact" >&2
  exit "$EVAL_STATUS"
fi
echo "=== completed duration-v2 job=${SLURM_JOB_ID} time=$(date -Is) ==="
