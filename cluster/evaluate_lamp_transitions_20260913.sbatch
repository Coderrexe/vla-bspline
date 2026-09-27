#!/bin/bash
# Submit afterok of both lamp training jobs. Prediction only, no robot access.
#SBATCH --job-name=lamp_transitions
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=r817u29n05,r817u23n05
#SBATCH --output=/gpfs/radev/home/zs442/vla_hardware_20260913/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
HW_ROOT="$HOME/vla_hardware_20260913"
export PYTHONPATH="$HW_ROOT/source/src:$HW_ROOT/transition_scripts:$HW_ROOT/scripts"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export HF_DATASETS_CACHE="$HW_ROOT/cache/datasets_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
for HW_RUN in lamp_assembling_spline_native_tcp_v1_2466234 lamp_assembling_waypoint_native_tcp_v1_2466235; do
  for HW_STEP in 010000 020000 030000 040000; do
    python "$HW_ROOT/transition_scripts/diagnose_apollo_lamp_transitions.py" evaluate \
      --inputs "$HW_ROOT/reports/lamp_transition_inputs_v1" \
      --videos "$HW_ROOT/raw/lamp_assembling" \
      --checkpoint "$HW_ROOT/outputs/$HW_RUN/checkpoints/$HW_STEP/pretrained_model" \
      --output "$HW_ROOT/reports/lamp_transitions_${SLURM_JOB_ID}/${HW_RUN}_${HW_STEP}"
  done
done
