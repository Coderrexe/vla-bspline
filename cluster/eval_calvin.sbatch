#!/bin/bash
#SBATCH --job-name=calv_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out
# usage: sbatch eval_calvin.sbatch <CKPT_DIR> [N_SEQ] [TAG] [SEQ_OFFSET] [N_TOTAL]
# two-process CALVIN eval: policy server (lerobot env, GPU) + sim client (calvin env)
# SEQ_OFFSET/N_TOTAL shard the deterministic official sequence list across jobs
set -uo pipefail
CKPT=$1
NSEQ=${2:-100}
TAG=${3:-eval}
OFFSET=${4:-0}
NTOTAL=${5:-0}
PORT=$((20000 + SLURM_JOB_ID % 20000))
OUTDIR=$HOME/scratch/vla_bspline/outputs
SRVLOG=$OUTDIR/calvin_server_${TAG}_${SLURM_JOB_ID}.log

module load miniconda
export HF_HOME=$HOME/scratch/vla_bspline/hf_cache TOKENIZERS_PARALLELISM=false

~/.conda/envs/lerobot/bin/python ~/vla_bspline/libero/calvin_policy_server.py \
  --ckpt "$CKPT" \
  --dataset_root $HOME/scratch/vla_bspline/calvin_v30 \
  --port $PORT > "$SRVLOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 60); do
  grep -q SERVER_READY "$SRVLOG" && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER DIED"; cat "$SRVLOG"; exit 1; }
  sleep 5
done
grep -q SERVER_READY "$SRVLOG" || { echo "SERVER TIMEOUT"; cat "$SRVLOG"; exit 1; }

~/.conda/envs/calvin/bin/python ~/vla_bspline/libero/calvin_eval_client.py \
  --debug_root $HOME/scratch/vla_bspline/calvin_debug_dataset/validation \
  --calvin_repo $HOME/vla_bspline/calvin \
  --port $PORT --n_seq $NSEQ --seq_offset $OFFSET --n_total $NTOTAL \
  --out $OUTDIR/calvin_eval_${TAG}.json
RC=$?
kill $SERVER_PID 2>/dev/null
tail -3 "$SRVLOG"
exit $RC
