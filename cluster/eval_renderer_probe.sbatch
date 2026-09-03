#!/bin/bash
#SBATCH --job-name=renderer_probe
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# LOCAL-ONLY REVIEW ARTIFACT. Do not submit until the renderer gate design is
# accepted and these sources are deliberately synced to the isolated SOURCE.
#
# Usage:
#   sbatch eval_renderer_probe.sbatch CKPT egl TAG REPLICATE
#   sbatch eval_renderer_probe.sbatch CKPT osmesa TAG REPLICATE
# Optional: STATES_PER_TASK=50 TASK_IDS=0,1 STATE_START=0 SEED_BASE=100000

set -euo pipefail

CKPT=${1:?expected checkpoint pretrained_model directory}
BACKEND=${2:?expected egl or osmesa}
TAG=${3:?expected filesystem-safe tag}
REPLICATE=${4:?expected nonnegative replicate index}
TASK_IDS=${TASK_IDS:-}
STATE_START=${STATE_START:-0}
STATES_PER_TASK=${STATES_PER_TASK:-10}
SEED_BASE=${SEED_BASE:-100000}
case "$BACKEND" in egl|osmesa) ;; *) echo "backend must be egl or osmesa" >&2; exit 2 ;; esac
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;; esac
for value in "$REPLICATE" "$STATE_START" "$STATES_PER_TASK" "$SEED_BASE"; do
  case "$value" in ''|*[!0-9]*) echo "invalid nonnegative integer: $value" >&2; exit 2 ;; esac
done
if [ "$STATES_PER_TASK" -le 0 ]; then
  echo "STATES_PER_TASK must be positive" >&2
  exit 2
fi
if [ -n "$TASK_IDS" ]; then
  case "$TASK_IDS" in *[!0-9,]*) echo "invalid TASK_IDS" >&2; exit 2 ;; esac
fi

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
if [ "$BACKEND" = osmesa ]; then
  module load Mesa/22.2.4-GCCcore-12.2.0
fi
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL="$BACKEND"
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
SOURCE=${RENDERER_PROBE_SOURCE_DIR:-$HOME/vla_bspline/cluster}
STAGE="$HOME/scratch/vla_bspline/code_snapshots/renderer_probe_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/renderer_probe"
OUT="$OUTDIR/${TAG}_${BACKEND}_r${REPLICATE}_${SLURM_JOB_ID}.json"
ARRAYS="$OUTDIR/${TAG}_${BACKEND}_r${REPLICATE}_${SLURM_JOB_ID}.npz"
test ! -e "$STAGE"
test ! -e "$OUT"
test ! -e "$ARRAYS"
mkdir -p "$STAGE" "$OUTDIR"
for name in \
  libero_renderer_probe.py \
  libero_hidden_state.py \
  libero_locked_eval_v2.py \
  libero_duration_ablation_v2.py \
  duration_intervention_v2.py \
  deterministic_eval.py
do
  cp "$SOURCE/$name" "$STAGE/$name"
done
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

ARGS=(
  --ckpt "$CKPT"
  --suite libero_10
  --state_start "$STATE_START"
  --states_per_task "$STATES_PER_TASK"
  --seed_base "$SEED_BASE"
  --control_freq 20
  --replicate_index "$REPLICATE"
  --out "$OUT"
  --arrays_out "$ARRAYS"
)
if [ -n "$TASK_IDS" ]; then
  ARGS+=(--task_ids "$TASK_IDS")
fi
python "$STAGE/libero_renderer_probe.py" "${ARGS[@]}"
test -s "$OUT"
test -s "$ARRAYS"

OUT="$OUT" BACKEND="$BACKEND" REPLICATE="$REPLICATE" python - <<'PY'
import json
import os

result = json.load(open(os.environ["OUT"]))
assert result["protocol"] == "libero_renderer_comparability_probe_v1"
assert result["renderer_backend"] == os.environ["BACKEND"]
assert result["replicate_index"] == int(os.environ["REPLICATE"])
assert result["n_episodes"] == len(result["episodes"])
assert result["opengl"]
PY
