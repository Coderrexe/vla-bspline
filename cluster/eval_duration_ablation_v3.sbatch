#!/bin/bash
#SBATCH --job-name=dur_stoch_v3
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Reviewed for the 2026-08-21 engineering smoke. Scientific pilot submission
# remains gated on the 30-rollout task-2 integration run.
#
# One process executes paired stochastic blocks. The defaults are the pilot:
# 12 states/task, one repeat, five episodes/block = 600 total episodes.
#
# Usage:
#   sbatch eval_duration_ablation_v3.sbatch CKPT TAG FIXED_MANIFEST.json
# Optional environment:
#   TASK_IDS=0,4 STATES_PER_TASK=12 STATE_START=0 REPEATS=1 SEED_BASE=100000

set -euo pipefail

CKPT=${1:?expected duration-enabled checkpoint pretrained_model directory}
TAG=${2:?expected filesystem-safe experiment tag}
FIXED_MANIFEST=${3:?expected complete fixed-duration training-prior manifest}
TASK_IDS=${TASK_IDS:-}
STATE_START=${STATE_START:-0}
STATES_PER_TASK=${STATES_PER_TASK:-12}
REPEATS=${REPEATS:-1}
SEED_BASE=${SEED_BASE:-100000}
ORDER_SEED=${ORDER_SEED:-20260821}
SHUFFLE_STRENGTH_THRESHOLD=${SHUFFLE_STRENGTH_THRESHOLD:-0.5}
REPEATABILITY_EQUIVALENCE_MARGIN=${REPEATABILITY_EQUIVALENCE_MARGIN:-0.10}
REQUIRE_EXACT_PREDICTED_REPLAY=${REQUIRE_EXACT_PREDICTED_REPLAY:-0}
case "$TAG" in
  *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;;
esac
for value in "$STATE_START" "$STATES_PER_TASK" "$REPEATS" "$SEED_BASE" "$ORDER_SEED"; do
  case "$value" in
    ''|*[!0-9]*) echo "invalid nonnegative integer: $value" >&2; exit 2 ;;
  esac
done
if [ "$STATES_PER_TASK" -le 0 ] || [ "$REPEATS" -le 0 ]; then
  echo "STATES_PER_TASK and REPEATS must be positive" >&2
  exit 2
fi
if [ "$REQUIRE_EXACT_PREDICTED_REPLAY" != 0 ] && [ "$REQUIRE_EXACT_PREDICTED_REPLAY" != 1 ]; then
  echo "REQUIRE_EXACT_PREDICTED_REPLAY must be 0 or 1" >&2
  exit 2
fi
if [ -n "$TASK_IDS" ]; then
  case "$TASK_IDS" in
    *[!0-9,]*) echo "TASK_IDS must be comma-separated nonnegative integers" >&2; exit 2 ;;
  esac
fi

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
if [ "${MUJOCO_GL:-egl}" = osmesa ]; then
  module load Mesa/22.2.4-GCCcore-12.2.0
fi
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
# EGL is the production default. A diagnostic allocation may explicitly set
# MUJOCO_GL=osmesa to test whether GPU rendering is the residual replay-noise
# source; the selected backend is captured in the runtime environment/log.
export MUJOCO_GL=${MUJOCO_GL:-egl}
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
SOURCE=${DURATION_SOURCE_DIR:-$HOME/vla_bspline/cluster}
test -d "$SOURCE"
STAGE="$HOME/scratch/vla_bspline/code_snapshots/duration_v3_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/duration_ablation_v3"
OUT="$OUTDIR/${TAG}_${SLURM_JOB_ID}.json"
test ! -e "$STAGE"
test ! -e "$OUT"
mkdir -p "$STAGE" "$OUTDIR"
for name in \
  libero_duration_ablation_v3.py \
  stochastic_duration_v3.py \
  duration_intervention_v3.py \
  libero_hidden_state.py \
  duration_intervention_v2.py \
  libero_duration_ablation_v2.py \
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
  --repeats "$REPEATS"
  --seed_base "$SEED_BASE"
  --order_seed "$ORDER_SEED"
  --control_freq 20
  --shuffle_seed 73921
  --shuffle_strength_threshold "$SHUFFLE_STRENGTH_THRESHOLD"
  --repeatability_equivalence_margin "$REPEATABILITY_EQUIVALENCE_MARGIN"
  --block_dir "$OUTDIR/${TAG}_blocks"
  --out "$OUT"
)
if [ -n "$TASK_IDS" ]; then
  ARGS+=(--task_ids "$TASK_IDS")
fi
if [ "$REQUIRE_EXACT_PREDICTED_REPLAY" = 1 ]; then
  ARGS+=(--require_exact_predicted_replay)
fi

echo "=== duration-v3 tag=${TAG} job=${SLURM_JOB_ID} node=$(hostname) ==="
python "$STAGE/libero_duration_ablation_v3.py" "${ARGS[@]}"
test -s "$OUT"

OUT="$OUT" STATES_PER_TASK="$STATES_PER_TASK" REPEATS="$REPEATS" python - <<'PY'
import json
import os

result = json.load(open(os.environ["OUT"]))
assert result["schema_version"] == 3
assert result["protocol"] == "libero_duration_stochastic_paired_block_v3"
assert result["stochastic_result"] is True
assert result["bitwise_replay_assumed"] is False
assert result["watermark"].startswith("STOCHASTIC_PAIRED_BLOCK_RESULT")
assert result["primary_predicted_arm"] == "predicted_eval"
assert result["scientific_arms"] == ["predicted_eval", "fixed", "shuffled"]
assert set(result["trace_only_arms"]) == {"predicted_donor", "predicted_closure"}
assert result["fixed_manifest_complete_training_scan"] is True
expected_blocks = (
    len(result["task_ids"])
    * int(os.environ["STATES_PER_TASK"])
    * int(os.environ["REPEATS"])
)
assert result["n_blocks"] == expected_blocks
assert result["n_episodes"] == expected_blocks * 5
expected_arms = {
    "predicted_donor", "predicted_eval", "fixed", "shuffled", "predicted_closure"
}
assert set(result["arm_summaries"]) == expected_arms
for summary in result["arm_summaries"].values():
    assert summary["n_episodes"] == expected_blocks
assert sum(result["efficacy_permutation_counts"].values()) == expected_blocks
for task in result["tasks"]:
    counts = list(task["task_order_manifest"]["permutation_counts"].values())
    assert sum(counts) == int(os.environ["STATES_PER_TASK"]) * int(os.environ["REPEATS"])
    assert max(counts) - min(counts) <= 1
    for block in task["blocks"]:
        assert block["arm_order"][0] == "predicted_donor"
        assert block["arm_order"][-1] == "predicted_closure"
        assert set(block["arm_order"][1:4]) == {"predicted_eval", "fixed", "shuffled"}
        assert block["shuffle_source_manifest"]["manifest_sha256"]
assert (
    result["n_blocks_executed_this_process"]
    + result["n_blocks_resumed_atomically"]
    == expected_blocks
)
assert result["order_balance_gate"]["passed"] is True
assert result["shuffle_strength_gate"]["target_minimum"] >= 0.5
assert set(result["predicted_repeatability_by_task"]) == {
    str(task_id) for task_id in result["task_ids"]
}
assert result["execution_runtime_manifests"]
if result["exact_predicted_replay_required"]:
    assert result["exact_predicted_replay_all_pairs"] is True
for block in (block for task in result["tasks"] for block in task["blocks"]):
    assert block["execution_runtime_sha256"] in result["execution_runtime_manifests"]
print(
    "validated stochastic paired blocks:", expected_blocks,
    "donor/closure agreement=",
    result["predicted_repeatability_all_pairs"]["donor_vs_closure"]["outcome_agreement_rate"],
)
PY

echo "=== completed duration-v3 job=${SLURM_JOB_ID} time=$(date -Is) ==="
