#!/bin/bash
# Frozen-model gate only. This is intentionally not a full-pilot launcher.
# It exercises every efficacy order once for the previously failing LIBERO-Long
# task IDs 3 and 8, one explicit state, six repeats.
# Usage: sbatch smoke_duration_ablation_v4_frozen_osmesa.sbatch CKPT TAG FIXED_MANIFEST
# Never submit this concurrently with the same TAG. Do not promote to a wider
# run unless this immutable output's frozen-initial-pairing gate passes.
#SBATCH --job-name=dur_v4_frozen_smoke
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:30:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
CKPT=${1:?expected duration-enabled pretrained_model directory}
TAG=${2:?expected filesystem-safe tag}
FIXED_MANIFEST=${3:?expected complete fixed-duration training-prior manifest}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag" >&2; exit 2;; esac
test -s "$CKPT/model.safetensors"
test -s "$CKPT/config.json"
test -s "$FIXED_MANIFEST"

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SOURCE=${DURATION_SOURCE_DIR:-$HOME/vla_bspline/scripts/eval}
test -d "$SOURCE"
STAGE="$HOME/scratch/vla_bspline/code_snapshots/duration_v4_frozen_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/duration_ablation_v4_frozen"
OUT="$OUTDIR/${TAG}_${SLURM_JOB_ID}.json"
test ! -e "$STAGE"
test ! -e "$OUT"
mkdir -p "$STAGE" "$OUTDIR"
for name in libero_duration_ablation_v4_frozen.py libero_frozen_reset.py duration_intervention_v3.py duration_intervention_v2.py libero_hidden_state.py libero_duration_ablation_v2.py libero_locked_eval_v2.py deterministic_eval.py stochastic_duration_v3.py
do
  cp "$SOURCE/$name" "$STAGE/$name"
done
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

# Inspection/static gate occurs before any simulator allocation.
python -m py_compile "$STAGE/libero_duration_ablation_v4_frozen.py" "$STAGE/libero_frozen_reset.py"
grep -q 'no_reset_after_block_capture' "$STAGE/libero_duration_ablation_v4_frozen.py"
grep -q 'mujoco.mj_setState' "$STAGE/libero_frozen_reset.py"

python "$STAGE/libero_duration_ablation_v4_frozen.py" \
  --ckpt "$CKPT" --fixed_manifest "$FIXED_MANIFEST" --suite libero_10 \
  --task_ids 3,8 --state_start 0 --states_per_task 1 --repeats 6 \
  --seed_base 100000 --order_seed 20260821 --control_freq 20 \
  --shuffle_seed 73921 --shuffle_strength_threshold 0.5 \
  --repeatability_equivalence_margin 0.10 --require_frozen_initial_pairing \
  --block_dir "$OUTDIR/${TAG}_blocks" --out "$OUT"

OUT="$OUT" python - <<'PY'
import json
import os

result = json.load(open(os.environ["OUT"]))
assert result["schema_version"] == 4
assert result["protocol"] == "libero_duration_stochastic_paired_block_v4_frozen_model"
assert result["frozen_model_paired_reset"] is True
assert result["no_reset_after_block_capture"] is True
assert result["terminal_autoreset_protocol"] == "suppress_libero_terminal_reset_and_restore_vector_next_step_flag_v1"
assert result["frozen_initial_pairing_required"] is True
assert result["frozen_initial_pairing_complete"] is True
assert result["task_ids"] == [3, 8]
assert result["state_ids"] == [0]
assert result["repeats"] == 6
assert result["n_blocks"] == 12
assert result["n_episodes"] == 60
assert result["order_balance_gate"]["passed"] is True
pairing = result["initial_pairing_summary"]
assert pairing["n_arm_episode_pairs"] == 48
for key, value in pairing.items():
    if key != "n_arm_episode_pairs":
        assert value == 48, (key, value)
for task in result["tasks"]:
    assert set(task["task_order_manifest"]["permutation_counts"].values()) == {1}
    for block in task["blocks"]:
        capture = block["frozen_capture"]
        for episode in block["episodes"].values():
            assert episode["initial_raw_observation_sha256"] == capture["initial_raw_observation_sha256"]
            assert episode["compiled_model_xml_sha256"] == capture["compiled_model_xml_sha256"]
            assert episode["frozen_restore_verified"] is True
            assert episode["terminal_autoreset_protocol"] == result["terminal_autoreset_protocol"]
print("passed v4 frozen-model initial pairing gate; full pilot remains prohibited")
PY

echo "completed frozen-model OSMesa smoke: $OUT"
