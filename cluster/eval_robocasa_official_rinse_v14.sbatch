#!/bin/bash
# Target-split official Rinse label/program pilot with videos.
# Usage: sbatch ... CKPT TAG env|rinse_official_fixed|rinse_official_oracle [N] [SEED]
#SBATCH --job-name=rc_official_rinse_eval
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out
set -euo pipefail
CKPT=${1:?checkpoint}; TAG=${2:?tag}; MODE=${3:?mode}; N=${4:-10}; SEED=${5:-1000}
case "$MODE" in env|rinse_official_fixed|rinse_official_oracle) ;; *) exit 64 ;; esac
case "$TAG" in *[!A-Za-z0-9_.-]*) exit 64 ;; esac
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
POLICY="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
EVAL="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_v14_official_rinse_20260822"
export PYTHONPATH="$POLICY/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_official_rinse_eval_v14"
VIDEO="$ROOT/videos/${TAG}_${MODE}_target_s${SEED}_n${N}_${SLURM_JOB_ID}"
OUT="$ROOT/${TAG}_${MODE}_target_s${SEED}_n${N}_${SLURM_JOB_ID}.json"
[[ -s "$CKPT/model.safetensors" && ! -e "$OUT" && ! -e "$VIDEO" ]] || exit 66
printf '%s\n' \
 "7768e31323b80b4900c66c005b635c99a8b5ed46743d05821e5a386894de19e7  $EVAL/robocasa_progress_eval.py" \
 "f1624203d4b954ca2a71827e3a5087a87e57f72195868af13dc09e693eae2b51  $EVAL/robocasa_video.py" \
 "8838b32d6a502a767a89847cf8fa0a146a4b03d30dd30f26d1301a8639a5b2d9  $EVAL/robocasa_language_programs.py" | sha256sum --check --strict
mkdir -p "$ROOT"
python "$EVAL/robocasa_progress_eval.py" --ckpt "$CKPT" --task RinseSinkBasin \
 --episodes "$N" --max_steps 1350 --seed_base "$SEED" --constructor_seed 20260821 \
 --split target --instr_mode "$MODE" --out "$OUT" --video_dir "$VIDEO" --video_stride 5
test -s "$OUT"
python - "$OUT" "$N" "$MODE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get("schema")!="robocasa_progress_eval_v14_official_rinse": raise RuntimeError("schema")
if d.get("n_episodes")!=int(sys.argv[2]) or d.get("instruction_mode")!=sys.argv[3]: raise RuntimeError("support")
if d.get("environment_split")!="target" or d.get("renderer")!="osmesa": raise RuntimeError("domain")
if len(d.get("video_artifacts",[]))!=int(sys.argv[2]): raise RuntimeError("videos")
print(f"successes={d['successes']}/{d['n_episodes']}")
PY
sha256sum "$OUT" "$VIDEO"/*.mp4
