#!/bin/bash
# Phase-conditioned spline timing: execute only Kettle placement at 0.5x
# control rate, with target-split OSMesa videos and immutable v13 source.
# Usage: sbatch ... CKPT TAG fixed|oracle [EPISODES] [SEED]
#SBATCH --job-name=rc_phase_timing
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail
CKPT=${1:?checkpoint required}
TAG=${2:?tag required}
SCHEDULER=${3:?fixed or oracle required}
EPISODES=${4:-10}
SEED=${5:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) exit 64 ;; esac
case "$SCHEDULER" in
  fixed) MODE=kettle_official_fixed_slowplace ;;
  oracle) MODE=kettle_official_oracle_slowplace ;;
  *) exit 64 ;;
esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POLICY_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
EVAL_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_v13_phase_timing_20260822"
export PYTHONPATH="$POLICY_SNAPSHOT/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$EVAL_SNAPSHOT/robocasa_progress_eval.py"
ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_official_kettle_eval_v13"
VIDEO="$ROOT/videos/${TAG}_${MODE}_target_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}"
OUT="$ROOT/${TAG}_${MODE}_target_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -s "$CKPT/model.safetensors" && -s "$CKPT/config.json" ]] || exit 66
[[ ! -e "$OUT" && ! -e "$VIDEO" ]] || exit 73
printf '%s\n' \
  "d90d8d64f1e413cc614691f98e8440c129771f25c88732acc847a82ba5da0f2a  $SCRIPT" \
  "f1624203d4b954ca2a71827e3a5087a87e57f72195868af13dc09e693eae2b51  $EVAL_SNAPSHOT/robocasa_video.py" \
  "3da90a9e372264a7724766d4d875497b09901aee7ef31dee231efacc88969599  $EVAL_SNAPSHOT/robocasa_language_programs.py" | sha256sum --check --strict
mkdir -p "$ROOT"

python "$SCRIPT" \
  --ckpt "$CKPT" --task KettleBoiling --episodes "$EPISODES" --max_steps 1500 \
  --seed_base "$SEED" --constructor_seed 20260821 --split target \
  --instr_mode "$MODE" --out "$OUT" --video_dir "$VIDEO" --video_stride 5

test -s "$OUT"
python - "$OUT" "$EPISODES" "$MODE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
if d.get("schema") != "robocasa_progress_eval_v13_phase_timing": raise RuntimeError("schema")
if d.get("renderer") != "osmesa" or d.get("environment_split") != "target": raise RuntimeError("domain")
if d.get("n_episodes") != int(sys.argv[2]) or d.get("instruction_mode") != sys.argv[3]: raise RuntimeError("support")
if d.get("phase_exec_rate_ratios") != [1.0, 2.0, 1.0]: raise RuntimeError("timing")
if len(d.get("video_artifacts", [])) != int(sys.argv[2]): raise RuntimeError("videos")
print(f"successes={d['successes']}/{d['n_episodes']}")
PY
sha256sum "$OUT" "$VIDEO"/*.mp4
