#!/bin/bash
# Prediction reports must have completed first. No robot access.
#SBATCH --job-name=lamp_handoff
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/radev/home/zs442/vla_hardware_20260913/logs/%x_%j.out
set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
HW_ROOT="$HOME/vla_hardware_20260913"
HW_PREVIOUS="$HOME/scratch/vla_bspline/hardware_20260912"
export PYTHONPATH="$HW_ROOT/transition_scripts:$HW_ROOT/scripts"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python "$HW_ROOT/transition_scripts/select_apollo_lamp_checkpoint.py" \
  --reports "$HW_ROOT/reports/lamp_transitions_2466263" \
  --inputs "$HW_ROOT/reports/lamp_transition_inputs_v1" \
  --output "$HW_ROOT/reports/lamp_checkpoint_selection_${SLURM_JOB_ID}.json"
for HW_NAME in training_source_v1.tar.gz HARDWARE_TRAINING_2026-09-12.md; do
  test ! -e "$HW_ROOT/manifests/$HW_NAME"
  cp -p "$HW_PREVIOUS/manifests/$HW_NAME" "$HW_ROOT/manifests/$HW_NAME"
done
python "$HW_ROOT/transition_scripts/package_apollo_handoff.py" \
  --root "$HW_ROOT" --output "$HW_ROOT/handoff_${SLURM_JOB_ID}" \
  --selection "$HW_ROOT/reports/lamp_checkpoint_selection_${SLURM_JOB_ID}.json" \
  --run lamp_assembling_spline_native_tcp_v1_2466234 \
  --run lamp_assembling_waypoint_native_tcp_v1_2466235
cp -p "$HW_ROOT/manifests/HARDWARE_CLOSEOUT_2026-09-13.md" "$HW_ROOT/handoff_${SLURM_JOB_ID}/"
cp -p "$HW_ROOT/manifests/LAMP_HANDOFF_README.md" "$HW_ROOT/handoff_${SLURM_JOB_ID}/"
