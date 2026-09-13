#!/bin/bash
#SBATCH --job-name=calv_clock
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=calv_clock-%j.out
# Usage: sbatch ... <uploaded source directory> <arm> <training seed> [nchains]
set -euo pipefail
SOURCE=$1
ARM=$2
SEED=$3
NCHAINS=${4:-50}
case "$ARM" in A_native|A_linear|C_native|C_retime) ;; *) exit 2 ;; esac
case "$SEED" in 1000|1001|1002) ;; *) exit 2 ;; esac
[[ "$NCHAINS" == 2 || "$NCHAINS" == 50 ]] || exit 2
BASE=/home/zs442/scratch/vla_bspline
RUN=$BASE/artifacts/calvin_matched_timing_20260905/${ARM}_s${SEED}_${SLURM_JOB_ID}
mkdir -p "$RUN"
cp -a /home/zs442/vla_bspline/lerobot/src "$RUN/lerobot_src"
cp -a "$SOURCE/scripts" "$RUN/scripts"
find "$RUN/scripts" "$RUN/lerobot_src" -type f -name '*.py' -exec sha256sum {} + > "$RUN/source_sha256.txt"
export PYTHONPATH="$RUN/lerobot_src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$BASE/hf_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
PY=/home/zs442/.conda/envs/lerobot/bin/python
CPY=/home/zs442/.conda/envs/calvin/bin/python
"$PY" "$RUN/scripts/analysis/check_waypoint_interpolation.py" > "$RUN/interpolation_gate.json"
SUFFIX=
[[ "$SEED" == 1000 ]] || SUFFIX=_s$SEED
HEAD=${ARM:0:1}
CKPT=$BASE/outputs/calv${HEAD}_100k${SUFFIX}/checkpoints/last/pretrained_model
"$PY" "$RUN/scripts/eval/prepare_calvin_timing_view.py" "$CKPT" "$RUN/checkpoint" --arm "$ARM"
PORT=$((20000 + SLURM_JOB_ID % 20000))
"$PY" "$RUN/scripts/eval/calvin_policy_server.py" --ckpt "$RUN/checkpoint" \
  --dataset_root "$BASE/calvin_v30" --port "$PORT" --matched_timing > "$RUN/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for ((attempt=0; attempt<90; attempt++)); do
  if grep -q SERVER_READY "$RUN/server.log"; then break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -40 "$RUN/server.log"; exit 1; }
  sleep 2
done
grep -q SERVER_READY "$RUN/server.log" || { tail -40 "$RUN/server.log"; exit 1; }
# Simulator uses its own pinned environment, not the lerobot PYTHONPATH.
PYTHONPATH= "$CPY" "$RUN/scripts/eval/calvin_eval_client.py" \
  --debug_root "$BASE/calvin_debug_dataset/validation" --calvin_repo /home/zs442/vla_bspline/calvin \
  --port "$PORT" --n_seq "$NCHAINS" --n_total 1000 --seq_offset 0 \
  --ep_len 360 --repeat 3 --policy_seed 700000 --exclusive --out "$RUN/result.json"
echo "RESULT $RUN/result.json"
