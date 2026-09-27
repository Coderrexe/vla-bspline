#!/bin/bash
#SBATCH --job-name=lamp_audit
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/home/zs442/vla_hardware_20260913/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
HW_ROOT="$HOME/vla_hardware_20260913"
HW_PREVIOUS="$HOME/scratch/vla_bspline/hardware_20260912"
test ! -e "$HW_ROOT/source"
cp -a "$HW_PREVIOUS/source" "$HW_ROOT/source"
cp -p "$HW_PREVIOUS/manifests/base_initialization.json" "$HW_ROOT/manifests/base_initialization.json"
export PYTHONPATH="$HW_ROOT/source/src:$HW_ROOT/scripts"
export HF_DATASETS_CACHE="$HW_ROOT/cache/datasets_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python "$HW_ROOT/scripts/audit_apollo_demos.py" --root "$HW_ROOT/raw" --out "$HW_ROOT/reports/lamp_audit"
python "$HW_ROOT/scripts/audit_apollo_pose_contract.py" --source "$HW_ROOT/raw/lamp_assembling" \
  --mjcf "$HW_ROOT/manifests/xarm7_on_rail.xml" --require corrected_tcp \
  --output "$HW_ROOT/reports/lamp_pose_contract.json"
python "$HW_ROOT/scripts/check_apollo_action_alignment.py" --source "$HW_ROOT/raw/lamp_assembling" \
  --output "$HW_ROOT/reports/lamp_action_alignment.json"
python - "$HW_ROOT" <<'PY'
import json, sys
from pathlib import Path
r=Path(sys.argv[1]); a=json.loads((r/'reports/lamp_audit/lamp_assembling_audit.json').read_text())
assert a['episode_count']==52 and a['frame_count']==23401
assert not a['issues'], a['issues']
PY
python "$HW_ROOT/scripts/export_apollo_lerobot.py" --source "$HW_ROOT/raw/lamp_assembling" \
  --output "$HW_ROOT/datasets/lamp_assembling_v1"
python "$HW_ROOT/scripts/gen_apollo_spline_stats.py" --dataset "$HW_ROOT/datasets/lamp_assembling_v1"
python - "$HW_ROOT" <<'PY'
import json, sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
r=Path(sys.argv[1]); root=r/'datasets/lamp_assembling_v1'
d=LeRobotDataset('bc_demo/lamp_assembling',root=root,
    delta_timestamps={'action':[i/25 for i in range(24)]}, video_backend='pyav')
for i in [0,len(d)//2,len(d)-1]:
    b=d[i]
    assert tuple(b['action'].shape)==(24,16)
    assert tuple(b['observation.state'].shape)==(32,)
    assert all(tuple(b[k].shape)==(3,480,640) for k in
               ['observation.images.grip_wrist','observation.images.view_wrist'])
contract={'schema_version':1,'state_pose_convention':'corrected_tcp',
          'action_pose_convention':'base_frame_delta_translation_and_spatial_rotvec',
          'gripper':'absolute_open_fraction_0_to_1',
          'audit':'reports/lamp_pose_contract.json','legacy_observation_conversion':False}
(root/'apollo_observation_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
(r/'manifests/lamp_dataset_ready.json').write_text(json.dumps({'reader':'PASS','task':'lamp_assembling',
     'train_episodes':47,'validation_episodes':5,'state_pose_convention':'corrected_tcp'})+'\n')
print('LAMP_DATASET_READY',flush=True)
PY
