#!/bin/bash
#SBATCH --job-name=dur_identity_seq
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Repeat the exact canonical evaluator twice, then run the predicted/no-op
# duration wrapper, sequentially inside one physical-GPU allocation. This
# distinguishes ordinary inference nondeterminism from wrapper-induced drift.

set -euo pipefail

CKPT=${1:?expected an n_action_steps=5 duration checkpoint}
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

SOURCE="$HOME/vla_bspline/cluster"
STAGE="$HOME/scratch/vla_bspline/code_snapshots/duration_identity_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/duration_identity"
test -s "$CKPT/model.safetensors"
test ! -e "$STAGE"
mkdir -p "$STAGE" "$OUTDIR"
for name in libero_locked_eval.py libero_duration_ablation.py duration_intervention.py; do
  cp "$SOURCE/$name" "$STAGE/$name"
done
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

RUNTIME="$STAGE/runtime.json"
RUNTIME="$RUNTIME" python - <<'PY'
import json
import os
import subprocess

import torch

query = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,uuid,pci.bus_id,driver_version",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip()
payload = {
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_smi": query,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": str(torch.backends.cudnn.version()),
}
open(os.environ["RUNTIME"], "w").write(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

CANON_A="$OUTDIR/canonical_a_${SLURM_JOB_ID}.json"
CANON_B="$OUTDIR/canonical_b_${SLURM_JOB_ID}.json"
HOOK="$OUTDIR/predicted_hook_${SLURM_JOB_ID}.json"
SUMMARY="$OUTDIR/identity_summary_${SLURM_JOB_ID}.json"
for path in "$CANON_A" "$CANON_B" "$HOOK" "$SUMMARY"; do test ! -e "$path"; done

for out in "$CANON_A" "$CANON_B"; do
  python "$STAGE/libero_locked_eval.py" \
    --ckpt "$CKPT" --suite libero_10 --state_start 0 --states_per_task 1 \
    --seed_base 100000 --control_freq 20 --out "$out"
done
python "$STAGE/libero_duration_ablation.py" \
  --ckpt "$CKPT" --suite libero_10 --state_start 0 --states_per_task 1 \
  --seed_base 100000 --control_freq 20 --duration_mode predicted --out "$HOOK"

CANON_A="$CANON_A" CANON_B="$CANON_B" HOOK="$HOOK" RUNTIME="$RUNTIME" \
SUMMARY="$SUMMARY" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(path):
    payload = json.load(open(path))
    return {
        (int(task["task_id"]), int(episode["state_id"])): {
            key: episode[key] for key in ("success", "steps", "policy_calls")
        }
        for task in payload["tasks"]
        for episode in task["episodes"]
    }


paths = {name: os.environ[name] for name in ("CANON_A", "CANON_B", "HOOK")}
records = {name: episodes(path) for name, path in paths.items()}
keys = sorted(records["CANON_A"])
if any(sorted(record) != keys for record in records.values()):
    raise RuntimeError("identity arms do not share one task/state key set")


def differences(left, right):
    return {
        f"{task}:{state}": {"left": left[(task, state)], "right": right[(task, state)]}
        for task, state in keys
        if left[(task, state)] != right[(task, state)]
    }


ab = differences(records["CANON_A"], records["CANON_B"])
ah = differences(records["CANON_A"], records["HOOK"])
result = {
    "schema_version": 1,
    "protocol": "single_gpu_sequential_duration_noop_identity_v1",
    "runtime": json.load(open(os.environ["RUNTIME"])),
    "inputs": {name: {"path": path, "sha256": digest(path)} for name, path in paths.items()},
    "canonical_repeat_exact": not ab,
    "predicted_hook_exact": not ah,
    "canonical_repeat_differences": ab,
    "predicted_hook_differences": ah,
}
temporary = Path(os.environ["SUMMARY"] + ".tmp")
temporary.write_text(json.dumps(result, indent=2) + "\n")
os.replace(temporary, os.environ["SUMMARY"])
print(json.dumps(result, indent=2))
PY

test -s "$SUMMARY"
echo "=== completed duration identity job=${SLURM_JOB_ID} time=$(date -Is) ==="
