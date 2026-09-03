#!/bin/bash
# Target-scene RoboCasa365 progress evaluation. Unlike legacy diagnostics,
# this explicitly binds split=target and therefore matches the target benchmark.
# Usage: sbatch ... CKPT TASK EPISODES MAX_STEPS TAG MODE [SEED]
#SBATCH --job-name=rc_target_v11
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail

CKPT=${1:?checkpoint required}
TASK=${2:?task required}
EPISODES=${3:?episode count required}
MAX_STEPS=${4:?max steps required}
TAG=${5:?tag required}
MODE=${6:-env}
SEED=${7:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 64 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

POLICY_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
EVAL_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_v11_target_20260822"
export PYTHONPATH="$POLICY_SNAPSHOT/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$EVAL_SNAPSHOT/robocasa_progress_eval.py"
OUT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_progress_v11_target"
SAFE_MODE=${MODE//:/_}
OUT="$OUT_ROOT/${TAG}_${TASK}_${SAFE_MODE}_target_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -f "$CKPT/model.safetensors" && -f "$CKPT/config.json" ]] || exit 66
[[ ! -e "$OUT" ]] || { echo "refusing overwrite" >&2; exit 73; }
printf '%s\n' \
  "9c7dd76eb240a935f5655719424c7799309880f2cb8ba10bb51437379d621a9d  $SCRIPT" \
  "f1624203d4b954ca2a71827e3a5087a87e57f72195868af13dc09e693eae2b51  $EVAL_SNAPSHOT/robocasa_video.py" \
  "2eca4bf5374257af8df1ec00d01685d6b355b63b375b0ae9c9a5841f5a6c30c7  $EVAL_SNAPSHOT/robocasa_language_programs.py" | sha256sum --check --strict
mkdir -p "$OUT_ROOT"

python "$SCRIPT" \
  --ckpt "$CKPT" --task "$TASK" --episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" --seed_base "$SEED" --constructor_seed 20260821 \
  --split target --instr_mode "$MODE" --out "$OUT"

test -s "$OUT"
python - "$OUT" "$EPISODES" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("schema") != "robocasa_progress_eval_v11_target_split":
    raise RuntimeError("unexpected result schema")
if result.get("renderer") != "osmesa" or result.get("environment_split") != "target":
    raise RuntimeError("result is not OSMesa/target bound")
if result.get("n_episodes") != int(sys.argv[2]):
    raise RuntimeError("incomplete result")
print(f"successes={result['successes']}/{result['n_episodes']} split=target")
PY
sha256sum "$SCRIPT" "$OUT"
