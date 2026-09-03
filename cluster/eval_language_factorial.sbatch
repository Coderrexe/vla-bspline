#!/bin/bash
#SBATCH --job-name=lang_eval
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Canonical paired LIBERO-Long evaluation. Each job creates an immutable
# checkpoint variant with an explicit replan cadence and evaluates an explicit
# state-id manifest (states 0..49, 500 episodes total). The evaluator forcibly
# sets and logs each state id, preventing outcome-dependent internal resets from
# silently changing the states visited by later episodes.
#
# Usage: sbatch eval_language_factorial.sbatch CKPT TAG N_ACTION_STEPS

set -euo pipefail

CKPT=${1:?expected checkpoint pretrained_model directory}
TAG=${2:?expected filesystem-safe experiment tag}
NAS=${3:?expected n_action_steps}
STATE_START=${STATE_START:-0}
STATES_PER_TASK=${STATES_PER_TASK:-50}
case "$TAG" in
  *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;;
esac
case "$NAS" in
  ''|*[!0-9]*) echo "invalid n_action_steps: $NAS" >&2; exit 2 ;;
esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable", file=sys.stderr)
    raise SystemExit(97)
print("gpu:", torch.cuda.get_device_name(0))
PY

test -s "$CKPT/model.safetensors"
VARIANT="$HOME/scratch/vla_bspline/outputs/overnight_variants/${TAG}_nas${NAS}_${SLURM_JOB_ID}"
OUT_DIR="$HOME/scratch/vla_bspline/outputs/overnight_eval"
OUT="$OUT_DIR/${TAG}_nas${NAS}_${SLURM_JOB_ID}.json"
test ! -e "$VARIANT"
test ! -e "$OUT"

CKPT="$CKPT" VARIANT="$VARIANT" NAS="$NAS" TAG="$TAG" STATE_START="$STATE_START" STATES_PER_TASK="$STATES_PER_TASK" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

src = Path(os.environ["CKPT"]).resolve()
dst = Path(os.environ["VARIANT"])
dst.mkdir(parents=True)
for item in src.iterdir():
    if item.name == "config.json":
        continue
    (dst / item.name).symlink_to(item.resolve())
cfg = json.loads((src / "config.json").read_text())
original_nas = cfg.get("n_action_steps")
cfg["n_action_steps"] = int(os.environ["NAS"])
(dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

manifest = {
    "tag": os.environ["TAG"],
    "source_checkpoint": str(src),
    "source_config_sha256": sha256(src / "config.json"),
    "model_sha256": sha256(src / "model.safetensors"),
    "original_n_action_steps": original_nas,
    "evaluated_n_action_steps": cfg["n_action_steps"],
    "policy_type": cfg["type"],
    "suite": "libero_10",
    "state_start": int(os.environ["STATE_START"]),
    "states_per_task": int(os.environ["STATES_PER_TASK"]),
    "control_frequency_hz": 20,
    "initial_states": list(
        range(
            int(os.environ["STATE_START"]),
            int(os.environ["STATE_START"]) + int(os.environ["STATES_PER_TASK"]),
        )
    ),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
}
(dst / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

echo "=== eval tag=${TAG} nas=${NAS} source=${CKPT} job=${SLURM_JOB_ID} node=$(hostname) ==="
python "$HOME/vla_bspline/cluster/libero_locked_eval.py" \
  --ckpt "$VARIANT" \
  --suite libero_10 \
  --state_start "$STATE_START" \
  --states_per_task "$STATES_PER_TASK" \
  --seed_base 100000 \
  --control_freq 20 \
  --out "$OUT"

test -s "$OUT"
OUT="$OUT" STATE_START="$STATE_START" STATES_PER_TASK="$STATES_PER_TASK" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["OUT"])
result = json.loads(path.read_text())
start = int(os.environ["STATE_START"])
count = int(os.environ["STATES_PER_TASK"])
assert result["n_episodes"] == 10 * count, result["n_episodes"]
assert result["state_ids"] == list(range(start, start + count)), result["state_ids"]
print("validated:", result["n_successes"], "/", result["n_episodes"])
PY

echo "=== completed tag=${TAG} nas=${NAS} job=${SLURM_JOB_ID} time=$(date -Is) ==="
