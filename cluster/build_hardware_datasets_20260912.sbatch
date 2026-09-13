#!/bin/bash
#SBATCH --job-name=hw_datasets
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/hardware_20260912/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export PYTHONPATH="$HOME/scratch/vla_bspline/hardware_20260912/source/src"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
HW_ROOT="$HOME/scratch/vla_bspline/hardware_20260912"
for task in cabinet_assembling drawer_assembling; do
  python "$HW_ROOT/scripts/export_apollo_lerobot.py" \
    --source "$HW_ROOT/raw/$task" --output "$HW_ROOT/datasets/${task}_v1"
done
python - <<'PY'
import json, os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = Path(os.environ['HOME'])/'scratch/vla_bspline/hardware_20260912'
for task in ['cabinet_assembling', 'drawer_assembling']:
    d = LeRobotDataset('bc_demo/'+task, root=root/'datasets'/f'{task}_v1',
                      delta_timestamps={'action': [i/25 for i in range(24)]}, video_backend='pyav')
    for i in [0, len(d)//2, len(d)-1]:
        b = d[i]
        for key in ['observation.state', 'action', 'observation.images.grip_wrist', 'observation.images.view_wrist']:
            print(task, i, key, tuple(b[key].shape), str(b[key].dtype), flush=True)
        assert tuple(b['action'].shape) == (24,16)
        assert tuple(b['observation.state'].shape) == (32,)
    print(task, 'LeRobot reader PASS', flush=True)
(root/'manifests/datasets_ready.json').write_text(json.dumps({'reader': 'PASS', 'tasks': ['cabinet_assembling','drawer_assembling']})+'\n')
PY
