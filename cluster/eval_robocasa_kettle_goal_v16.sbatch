#!/bin/bash
# Target-split evaluation for burner-goal-consistent Kettle language.
# Usage: sbatch ... CKPT TAG kettle_goal_fixed|kettle_goal_oracle [N] [SEED] [BURNER]
# BURNER=auto selects the most frequent officially labeled burner available in
# the static fixture; this is fixed before rollout and never uses progress.
#SBATCH --job-name=rc_kettle_goal_v16
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out
set -euo pipefail

CKPT=${1:?checkpoint required}; TAG=${2:?tag required}; MODE=${3:?mode required}
N=${4:-10}; SEED=${5:-1000}; BURNER=${6:-auto}
case "$TAG" in *[!A-Za-z0-9_.-]*) exit 64 ;; esac
case "$MODE" in kettle_goal_fixed|kettle_goal_oracle) ;; *) exit 64 ;; esac
case "$BURNER" in auto|center|front-center|front-left|front-right|left|rear-left|rear-right) ;; *) exit 64 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POLICY="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
EVAL="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_v16_goal_consistent_20260822"
export PYTHONPATH="$POLICY/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$EVAL/robocasa_progress_eval.py"
ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_kettle_goal_eval_v16"
VIDEO="$ROOT/videos/${TAG}_${MODE}_${BURNER}_target_s${SEED}_n${N}_${SLURM_JOB_ID}"
OUT="$ROOT/${TAG}_${MODE}_${BURNER}_target_s${SEED}_n${N}_${SLURM_JOB_ID}.json"
test -s "$CKPT/model.safetensors"; test -s "$CKPT/config.json"
test ! -e "$OUT"; test ! -e "$VIDEO"
printf '%s\n' \
  "b925987859b2e0dc070c13d1a5b264284a9d8463503028458bee472aa9182ff2  $SCRIPT" \
  "89f9f382b4a75b676c9b931d785e023c288dec664f5a2f8b96baadaebecc5e7b  $EVAL/robocasa_language_programs.py" \
  "f1624203d4b954ca2a71827e3a5087a87e57f72195868af13dc09e693eae2b51  $EVAL/robocasa_video.py" | sha256sum --check --strict
mkdir -p "$ROOT"

python "$SCRIPT" --ckpt "$CKPT" --task KettleBoiling --episodes "$N" \
  --max_steps 1500 --seed_base "$SEED" --constructor_seed 20260821 --split target \
  --instr_mode "$MODE" --kettle_goal_burner "$BURNER" --out "$OUT" \
  --video_dir "$VIDEO" --video_stride 5

python - "$OUT" "$N" "$MODE" "$BURNER" <<'PY'
import json,re,sys
d=json.load(open(sys.argv[1]))
if d.get("schema") != "robocasa_progress_eval_v16_goal_consistent_kettle": raise RuntimeError("schema")
if d.get("renderer") != "osmesa" or d.get("environment_split") != "target": raise RuntimeError("domain")
if d.get("n_episodes") != int(sys.argv[2]) or d.get("instruction_mode") != sys.argv[3]: raise RuntimeError("support")
if d.get("kettle_goal_burner_request") != sys.argv[4]: raise RuntimeError("burner request")
if len(d.get("video_artifacts", [])) != int(sys.argv[2]): raise RuntimeError("videos")
for episode in d["episodes"]:
    location=episode["kettle_goal_location"]
    prompts=episode["instruction_sentences"]
    if prompts != [
        "pick up the kettle from the counter",
        f"place the kettle on the {location} stove burner",
        f"turn on the {location} burner where the kettle is placed",
    ]: raise RuntimeError(f"goal inconsistency: {prompts}")
    for key in ("selected_burner_xy_distance", "selected_burner_on"):
        if key not in episode["extrema"]: raise RuntimeError(f"missing {key}")
print(f"successes={d['successes']}/{d['n_episodes']}")
PY
sha256sum "$OUT" "$VIDEO"/*.mp4
