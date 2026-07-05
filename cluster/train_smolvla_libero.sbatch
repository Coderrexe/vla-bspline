#!/bin/bash
#SBATCH --job-name=smolvla_libero
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# Stock SmolVLA -> LIBERO baseline (Baseline A) via LeRobot.
# Override knobs with:  sbatch --export=ALL,STEPS=2000,EVAL_EP=1,TASK=libero_10 train_smolvla_libero.sbatch
# Default here is a SHORT SMOKE TEST; bump STEPS to 100000 for the real baseline.

set -uo pipefail
echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load miniconda
source activate $HOME/.conda/envs/lerobot

# headless rendering for the LIBERO / MuJoCo simulator
export MUJOCO_GL=egl
# keep dataset/model/checkpoint I/O on scratch (home quota is tight from the finance project)
export HF_HOME=$HOME/scratch_pi_zy286/zs442/vla_bspline/hf_cache
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

STEPS=${STEPS:-2000}
TASK=${TASK:-libero_10}
EVAL_EP=${EVAL_EP:-1}
BATCH=${BATCH:-4}
OUT=$HOME/scratch_pi_zy286/zs442/vla_bspline/outputs/smolvla_libero_${SLURM_JOB_ID}

echo "=== STEPS=$STEPS TASK=$TASK EVAL_EP=$EVAL_EP BATCH=$BATCH OUT=$OUT ==="

lerobot-train \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=${TASK} \
  --output_dir=${OUT} \
  --steps=${STEPS} \
  --batch_size=${BATCH} \
  --save_freq=$(( STEPS < 2000 ? STEPS : 2000 )) \
  --eval_freq=${STEPS} \
  --eval.batch_size=1 \
  --eval.n_episodes=${EVAL_EP} \
  --wandb.enable=false

echo "=== train exit: $? | $(date) ==="
