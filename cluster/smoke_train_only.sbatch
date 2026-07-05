#!/bin/bash
#SBATCH --job-name=smolvla_smoke
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# gpu_devel: ~no queue, 6h cap, 2 GPUs / 256G / 1 job max per user.
# TRAIN-ONLY smoke test: validate SmolVLA trains on the LIBERO LeRobot dataset on GPU,
# independent of the LIBERO simulator (eval) dependency (egl_probe still being fixed).
# Downloads: HuggingFaceVLA/libero dataset + smolvla_base VLM weights, then ~200 steps.

set -uo pipefail
echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

module load miniconda
source activate $HOME/.conda/envs/lerobot

# keep dataset/model/checkpoint I/O on scratch (home quota is tight from the finance project)
export HF_HOME=$HOME/scratch_pi_zy286/zs442/vla_bspline/hf_cache
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# quick CUDA sanity on the GPU node
python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

OUT=$HOME/scratch_pi_zy286/zs442/vla_bspline/outputs/smoke_${SLURM_JOB_ID}
lerobot-train \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --output_dir=${OUT} \
  --steps=200 \
  --batch_size=4 \
  --num_workers=2 \
  --save_freq=200 \
  --log_freq=25 \
  --wandb.enable=false

echo "=== train exit: $? | $(date) ==="
ls -la ${OUT} 2>&1 | head
