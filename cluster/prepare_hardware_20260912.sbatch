#!/bin/bash
# Download the shared pretrained initialization and inspect the pinned runtime.
#SBATCH --job-name=hw_prepare
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/hardware_20260912/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export PYTHONPATH="$HOME/scratch/vla_bspline/hardware_20260912/source/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
python - <<'PY'
import importlib.metadata as m
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
root = Path(os.environ['HOME']) / 'scratch/vla_bspline/hardware_20260912'
repo = 'lerobot/smolvla_base'
revision = HfApi().model_info(repo).sha
path = snapshot_download(repo, revision=revision)
payload = {'base_repo': repo, 'revision': revision, 'snapshot': path,
           'python': sys.version, 'packages': {}}
for name in ['torch', 'torchvision', 'transformers', 'lerobot', 'datasets',
             'safetensors', 'av', 'torchcodec', 'huggingface_hub', 'accelerate']:
    try: payload['packages'][name] = m.version(name)
    except m.PackageNotFoundError: payload['packages'][name] = None
(root/'manifests/base_initialization.json').write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps(payload, indent=2), flush=True)
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('Imports OK', flush=True)
PY
