#!/bin/bash
# Resume a preempted or timed-out focused RoboCasa curriculum run.
# Usage: sbatch resume_robocasa_curriculum.sbatch OUTPUT_DIR
#SBATCH --job-name=rc_curr_resume
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
OUT=${1:?expected exact curriculum output directory}
case "$(basename "$OUT")" in
  rccurr_easy4_task_C_s*_*) ;;
  rccurr_easy4_gran_C_s*_*) ;;
  *) echo "refusing unexpected output path: $OUT" >&2; exit 64 ;;
esac
CONFIG="$OUT/checkpoints/last/pretrained_model/train_config.json"
test -s "$CONFIG"

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(97)
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY

python -m lerobot.scripts.lerobot_train --config_path="$CONFIG" --resume=true
test -s "$OUT/checkpoints/last/pretrained_model/model.safetensors"
echo "completed resume output=$OUT"
