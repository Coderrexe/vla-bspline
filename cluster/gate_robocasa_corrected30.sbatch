#!/bin/bash
# Validate an immutable corrected-contract RoboCasa 30k checkpoint, then submit
# low-priority renderer-controlled n=5 diagnostics.  This gate never imports or
# edits the installed spline policy; all contract inputs come from the frozen
# corrected snapshot.
#
# Usage: sbatch gate_robocasa_corrected30.sbatch task|gran TRAIN_JOB_ID OUTPUT
#SBATCH --job-name=rc_corr30_gate
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

ARM=${1:?expected task or gran}
TRAIN_JOB=${2:?training job id required}
OUT=${3:?exact corrected curriculum output required}
case "$ARM" in
  task) EXPECTED_BASE="rccorrected_easy4_task_C_s1000_100000_${TRAIN_JOB}" ;;
  gran) EXPECTED_BASE="rccorrected_easy4_gran_C_s1000_100000_${TRAIN_JOB}" ;;
  *) echo "unsupported arm: $ARM" >&2; exit 64 ;;
esac
[[ "$(basename "$OUT")" == "$EXPECTED_BASE" ]] || {
  echo "output/job/arm mismatch: $OUT" >&2
  exit 64
}

SNAP="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
POLICY_DIR="$SNAP/src/lerobot/policies/smolvla_spline"
STATS="$POLICY_DIR/spline_stats_robocasa_composite_seen_all_v2_n8h24.json"
EVALUATOR="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_eval_v9_20260821/robocasa_progress_eval.py"
EVAL_LAUNCHER="$HOME/vla_bspline/cluster/eval_robocasa_progress_v9_osmesa.sbatch"
TRAIN_LOG="$HOME/vla_bspline/cluster/rc_corr4-${TRAIN_JOB}.out"
TRAIN_MANIFEST="$HOME/scratch/vla_bspline/manifests/robocasa_corrected_curriculum/${EXPECTED_BASE}.json"
CKPT="$OUT/checkpoints/030000/pretrained_model"
STATE="$OUT/checkpoints/030000/training_state/training_step.json"
RESULT_DIR="$HOME/scratch/vla_bspline/manifests/robocasa_corrected_eval_gate"
RESULT="$RESULT_DIR/${EXPECTED_BASE}_30k_gate_${SLURM_JOB_ID}.json"
CLAIM="$RESULT_DIR/${EXPECTED_BASE}_30k.claim"

module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"

# A 30k save takes roughly three hours from training start.  Starting this gate
# shortly before that point is expected; wait only while the named job remains
# live, and never accept a partially written checkpoint.
deadline=$((SECONDS + 10200))
while [[ ! -s "$CKPT/model.safetensors" || ! -s "$CKPT/config.json" || ! -s "$STATE" ]]; do
  state=$(sacct -j "$TRAIN_JOB" -X -n -o State | awk 'NF {print $1; exit}')
  case "$state" in
    RUNNING|PENDING) ;;
    *) echo "training job became terminal before a complete 30k save: $state" >&2; exit 1 ;;
  esac
  (( SECONDS < deadline )) || { echo "timed out waiting for 30k checkpoint" >&2; exit 1; }
  sleep 45
done

for FILE in "$STATS" "$EVALUATOR" "$EVAL_LAUNCHER" "$TRAIN_LOG" \
  "$TRAIN_MANIFEST" "$CKPT/model.safetensors" "$CKPT/config.json" "$STATE"; do
  [[ -s "$FILE" ]] || { echo "missing gate input: $FILE" >&2; exit 66; }
done

check_sha() {
  local expected=$1 path=$2 observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || {
    echo "SHA mismatch for $path: $observed" >&2
    exit 1
  }
}

check_sha 287be4152eb77bd9301a8d62e26123574356f9cb543e1b8050406e783ec78007 "$STATS"
check_sha 9335cae53e277aa76975d59f43c266b0a1beee17d12d7737fd2d09db3016e9f6 "$POLICY_DIR/event_targets.py"
check_sha 86cb9533bb2efa1172139c12e9e7fe5a13065f82ad7a5f807a983d50faf373bc "$POLICY_DIR/modeling_smolvla_spline.py"
check_sha 1981c0fe995a046c4fa437317f4983fe36c811505bbdc4ac51105847b86de8aa "$POLICY_DIR/configuration_smolvla_spline.py"
check_sha efcedf1c3d86e84bda83660c53ffdd61ffece9a7b93088ec8445dfa1fc9dc952 "$POLICY_DIR/stats_contract.py"
check_sha 89ca38b043538c8bb932fea7e2bbc391ae12957e5a6aa7eb30f5b9b5ebf64679 "$EVALUATOR"
check_sha 376a4b82d316862c6c68d6440fa193c3c3eb266ea0c27642315fb11ba982dc61 "$EVAL_LAUNCHER"

# Protect the legacy installed policy used by older jobs.  Corrected checkpoints
# must remain isolated instead of silently changing that mutable installation.
INSTALLED="$HOME/vla_bspline/lerobot/src/lerobot/policies/smolvla_spline"
check_sha bd1e0fc2f8b2d274946c4e87f898cd80a154fac9c59e4f2280a5332789a03f11 "$INSTALLED/configuration_smolvla_spline.py"
check_sha 4315fe7a4d761a46cf0580a034208fad2b38dfddf770b7630f8dd18fda4d474d "$INSTALLED/modeling_smolvla_spline.py"
check_sha bbc65e993e96a205469f27877017aa56adc9f2fd79fc6920e48f86cdf9109d49 "$INSTALLED/spline_stats_robocasa_comp_v2_n8h24.json"

python - "$ARM" "$TRAIN_JOB" "$OUT" "$TRAIN_MANIFEST" "$TRAIN_LOG" \
  "$CKPT" "$STATE" "$STATS" <<'PY'
import json
import math
import os
import re
import sys

import numpy as np
from safetensors import safe_open

arm, train_job, output, manifest_path, log_path, ckpt, state_path, stats_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
expected_sources = {
    "configuration": "1981c0fe995a046c4fa437317f4983fe36c811505bbdc4ac51105847b86de8aa",
    "corrected_stats": "287be4152eb77bd9301a8d62e26123574356f9cb543e1b8050406e783ec78007",
    "event_targets": "9335cae53e277aa76975d59f43c266b0a1beee17d12d7737fd2d09db3016e9f6",
    "modeling": "86cb9533bb2efa1172139c12e9e7fe5a13065f82ad7a5f807a983d50faf373bc",
    "stats_contract": "efcedf1c3d86e84bda83660c53ffdd61ffece9a7b93088ec8445dfa1fc9dc952",
}
assert manifest["schema"] == "robocasa_corrected_curriculum_train_v1"
assert manifest["job_id"] == train_job and manifest["arm"] == arm
assert manifest["seed"] == 1000 and manifest["steps"] == 100000
assert manifest["episode_count"] == 2029
assert manifest["selected_physical_transition_sha256"] == \
    "e0ec48daadc20518f9f42f7b61d074d1f2347abe83a669b7dd5a7811e7d88815"
assert manifest["initial_model_sha256"] == \
    "42c5fcaa07738005f8cf30b05d3f5432bb549857735d059d4292e0694c2e3f33"
assert manifest["source_sha256"] == expected_sources
assert manifest["normalization_contract"] == {
    "checkpoint_embedding_required": True,
    "event_boundary_semantics": "production_event_index_k_exclusive",
    "stats_sha256": expected_sources["corrected_stats"],
}
assert os.path.realpath(manifest["output"]) == os.path.realpath(output)

state = json.load(open(state_path, encoding="utf-8"))
assert state["step"] == 30000 and state["num_processes"] == 1 and state["batch_size"] == 32

config = json.load(open(os.path.join(ckpt, "config.json"), encoding="utf-8"))
stats = json.load(open(stats_path, encoding="utf-8"))
embedded = config.get("embedded_spline_stats")
assert embedded and all(embedded.get(key) == value for key, value in stats.items())
assert config["action_layout"] == "robocasa12" and config["predict_duration"] is True
assert (config["n_ctrl"], config["min_seg"], config["horizon_max"]) == (8, 8, 24)
expected_extra = {
    "degree": 3,
    "action_layout": "robocasa12",
    "pose_dims": [5, 6, 7, 8, 9, 10],
    "grip_idx": 11,
    "pass_dims": [0, 1, 2, 3, 4],
    "boundary_semantics": "production_event_index_k_exclusive",
    "stats_contract_version": 1,
}
assert all(embedded.get(key) == value for key, value in expected_extra.items())

mean = np.zeros((8, 13), dtype=np.float32)
std = np.ones((8, 13), dtype=np.float32)
mean[:, :6] = embedded["pose_ctrl_mean"]
std[:, :6] = embedded["pose_ctrl_std"]
mean[:, 6] = embedded["grip_ctrl_mean"]
std[:, 6] = embedded["grip_ctrl_std"]
mean[:, 7] = embedded["logT_mean"]
std[:, 7] = embedded["logT_std"]
mean[:, 8:] = embedded["pass_ctrl_mean"]
std[:, 8:] = embedded["pass_ctrl_std"]
with safe_open(os.path.join(ckpt, "model.safetensors"), framework="numpy") as tensors:
    observed_mean = tensors.get_tensor("_tgt_mean")
    observed_std = tensors.get_tensor("_tgt_std")
assert np.array_equal(observed_mean, mean) and np.array_equal(observed_std, std)

pattern = re.compile(r"loss:(\S+) grdn:(\S+).*updt_s:(\S+).*smp/s:(\S+)")
health = []
for line in open(log_path, encoding="utf-8", errors="replace"):
    match = pattern.search(line)
    if match:
        health.append(tuple(float(value) for value in match.groups()))
assert len(health) >= 250
assert all(all(math.isfinite(value) for value in row) for row in health)
assert all(row[0] >= 0 and row[1] >= 0 and row[2] > 0 and row[3] > 0 for row in health)
print(json.dumps({
    "arm": arm,
    "checkpoint": os.path.realpath(ckpt),
    "step": state["step"],
    "embedded_stats_exact": True,
    "normalization_tensors_exact": True,
    "health_records": len(health),
    "latest_health": dict(zip(("loss", "grad_norm", "update_s", "samples_s"), health[-1])),
}, sort_keys=True))
PY

# Reserve an exclusive result path before submission so a duplicate gate cannot
# race this one.  The evaluation jobs are deliberately de-prioritized relative
# to older and teammate work.
mkdir -p "$RESULT_DIR"
set -o noclobber
: > "$CLAIM"
set +o noclobber

KETTLE_JOB=$(sbatch --parsable --nice=10000 "$EVAL_LAUNCHER" \
  "$CKPT" KettleBoiling 5 1500 "corr${ARM}30OSM_env" env 1000)
RINSE_JOB=$(sbatch --parsable --nice=10000 "$EVAL_LAUNCHER" \
  "$CKPT" RinseSinkBasin 5 1350 "corr${ARM}30OSM_env" env 1000)

python - "$RESULT" "$ARM" "$TRAIN_JOB" "$CKPT" "$KETTLE_JOB" "$RINSE_JOB" <<'PY'
import hashlib
import json
import os
import sys

path, arm, train_job, ckpt, kettle_job, rinse_job = sys.argv[1:]
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema": "robocasa_corrected_30k_eval_gate_v1",
    "gate_job_id": os.environ["SLURM_JOB_ID"],
    "arm": arm,
    "training_job_id": train_job,
    "checkpoint": os.path.realpath(ckpt),
    "checkpoint_model_sha256": sha256(os.path.join(ckpt, "model.safetensors")),
    "checkpoint_config_sha256": sha256(os.path.join(ckpt, "config.json")),
    "contract_validated": True,
    "installed_policy_untouched": True,
    "diagnostic_only": True,
    "n_per_task": 5,
    "constructor_seed": 20260821,
    "episode_seed_base": 1000,
    "renderer": "osmesa",
    "instruction_mode": "env",
    "eval_jobs": {"KettleBoiling": kettle_job, "RinseSinkBasin": rinse_job},
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "validated corrected $ARM 30k; submitted Kettle=$KETTLE_JOB Rinse=$RINSE_JOB"
