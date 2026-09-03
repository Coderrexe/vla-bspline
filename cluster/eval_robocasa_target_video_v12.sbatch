#!/bin/bash
# Exact-seed target-split video replay for RoboCasa failure/success forensics.
# Usage: sbatch ... CKPT TASK TAG SEED [MODE]
#SBATCH --job-name=rc_target_video
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail
CKPT=${1:?checkpoint}
TASK=${2:?task}
TAG=${3:?tag}
SEED=${4:?seed}
MODE=${5:-env}
case "$TASK" in KettleBoiling) MAX_STEPS=1500 ;; RinseSinkBasin) MAX_STEPS=1350 ;; *) exit 64 ;; esac
case "$TAG" in *[!A-Za-z0-9_.-]*) exit 64 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POLICY_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
EVAL_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_v12_official_20260822"
export PYTHONPATH="$POLICY_SNAPSHOT/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$EVAL_SNAPSHOT/robocasa_progress_eval.py"
ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_target_video_v12"
VIDEO="$ROOT/videos/${TAG}_${TASK}_${MODE}_target_s${SEED}_${SLURM_JOB_ID}"
OUT="$ROOT/${TAG}_${TASK}_${MODE}_target_s${SEED}_${SLURM_JOB_ID}.json"
[[ -s "$CKPT/model.safetensors" && -s "$CKPT/config.json" ]] || exit 66
[[ ! -e "$OUT" && ! -e "$VIDEO" ]] || exit 73
printf '%s\n' \
  "5b73d27bef9c82455f9c2ec74a4710d93aea688f5bd066aa7d4f4fc4d85ab656  $SCRIPT" \
  "f1624203d4b954ca2a71827e3a5087a87e57f72195868af13dc09e693eae2b51  $EVAL_SNAPSHOT/robocasa_video.py" \
  "ffca203b363ddfd0055ea7f6e0b9c9e4de55c20d15f9bd57868fe205ba9af5fc  $EVAL_SNAPSHOT/robocasa_language_programs.py" | sha256sum --check --strict
mkdir -p "$ROOT"
python "$SCRIPT" \
  --ckpt "$CKPT" --task "$TASK" --episodes 1 --max_steps "$MAX_STEPS" \
  --seed_base "$SEED" --constructor_seed 20260821 --split target \
  --instr_mode "$MODE" --out "$OUT" --video_dir "$VIDEO" --video_stride 2
test -s "$OUT"
python - "$OUT" "$TASK" "$SEED" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
if d.get("schema") != "robocasa_progress_eval_v12_official_program": raise RuntimeError("schema")
if d.get("task") != sys.argv[2] or d.get("seed_base") != int(sys.argv[3]): raise RuntimeError("support")
if d.get("environment_split") != "target" or d.get("renderer") != "osmesa": raise RuntimeError("domain")
if len(d.get("video_artifacts", [])) != 1: raise RuntimeError("video")
print(f"successes={d['successes']}/1")
PY
sha256sum "$OUT" "$VIDEO"/*.mp4
