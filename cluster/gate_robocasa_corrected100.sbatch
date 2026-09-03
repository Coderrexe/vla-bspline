#!/bin/bash
# Fail-closed final-checkpoint gate for the corrected-contract RoboCasa run.
# It schedules renderer-controlled n=10 diagnostics only after the actual 100k
# checkpoint, embedded target contract, and frozen-source hashes validate.
# Usage: sbatch gate_robocasa_corrected100.sbatch task|gran TRAIN_JOB OUTPUT
# OUTPUT may originate from an earlier training job when TRAIN_JOB is a resume.
#
# This intentionally does not edit the installed policy or promote an n=10
# result to a paper claim. It is the preregistered decision point for whether
# a larger, distributional evaluation is warranted.
#SBATCH --job-name=rc_corr100_gate
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out

set -euo pipefail

ARM=${1:?expected task or gran}
TRAIN_JOB=${2:?training job id required}
OUT=${3:?corrected training output required}
case "$ARM" in task|gran) ;; *) echo "unsupported arm: $ARM" >&2; exit 64 ;; esac
case "$(basename "$OUT")" in rccorrected_easy4_"$ARM"_C_s1000_100000_*) ;; *)
  echo "output/arm mismatch: $OUT" >&2; exit 64 ;;
esac

SNAP="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
POLICY_DIR="$SNAP/src/lerobot/policies/smolvla_spline"
STATS="$POLICY_DIR/spline_stats_robocasa_composite_seen_all_v2_n8h24.json"
EVALUATOR="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_eval_v9_20260821/robocasa_progress_eval.py"
EVAL_LAUNCHER="$HOME/vla_bspline/cluster/eval_robocasa_progress_v9_osmesa.sbatch"
CKPT="$OUT/checkpoints/100000/pretrained_model"
STATE="$OUT/checkpoints/100000/training_state/training_step.json"
TRAIN_MANIFEST="$HOME/scratch/vla_bspline/manifests/robocasa_corrected_curriculum/$(basename "$OUT").json"
MANIFEST_DIR="$HOME/scratch/vla_bspline/manifests/robocasa_corrected_final_eval_gate"
RESULT="$MANIFEST_DIR/$(basename "$OUT")_100k_gate_${SLURM_JOB_ID}.json"
CLAIM="$MANIFEST_DIR/$(basename "$OUT")_100k.claim"

for path in "$SNAP" "$STATS" "$EVALUATOR" "$EVAL_LAUNCHER" "$CKPT/model.safetensors" "$CKPT/config.json" "$STATE" "$TRAIN_MANIFEST"; do
  [[ -s "$path" || -d "$path" ]] || { echo "missing final-gate input: $path" >&2; exit 66; }
done

check_sha() {
  local expected=$1 path=$2 got
  got=$(sha256sum "$path" | awk '{print $1}')
  [[ "$got" == "$expected" ]] || { echo "SHA mismatch for $path: $got" >&2; exit 1; }
}
check_sha 287be4152eb77bd9301a8d62e26123574356f9cb543e1b8050406e783ec78007 "$STATS"
check_sha 9335cae53e277aa76975d59f43c266b0a1beee17d12d7737fd2d09db3016e9f6 "$POLICY_DIR/event_targets.py"
check_sha 86cb9533bb2efa1172139c12e9e7fe5a13065f82ad7a5f807a983d50faf373bc "$POLICY_DIR/modeling_smolvla_spline.py"
check_sha 89ca38b043538c8bb932fea7e2bbc391ae12957e5a6aa7eb30f5b9b5ebf64679 "$EVALUATOR"

module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
python - "$OUT" "$CKPT" "$STATE" "$STATS" "$TRAIN_MANIFEST" "$ARM" <<'PY'
import json, math, os, sys
import numpy as np
from safetensors import safe_open

output, ckpt, state_path, stats_path, manifest_path, arm = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
assert manifest["schema"] == "robocasa_corrected_curriculum_train_v1"
assert manifest["arm"] == arm and manifest["seed"] == 1000 and manifest["steps"] == 100000
assert manifest["episode_count"] == 2029
assert manifest["selected_physical_transition_sha256"] == "e0ec48daadc20518f9f42f7b61d074d1f2347abe83a669b7dd5a7811e7d88815"
assert manifest["initial_model_sha256"] == "42c5fcaa07738005f8cf30b05d3f5432bb549857735d059d4292e0694c2e3f33"
assert manifest["normalization_contract"]["event_boundary_semantics"] == "production_event_index_k_exclusive"
assert manifest["normalization_contract"]["stats_sha256"] == "287be4152eb77bd9301a8d62e26123574356f9cb543e1b8050406e783ec78007"
state = json.load(open(state_path, encoding="utf-8"))
assert state["step"] == 100000 and state["num_processes"] == 1 and state["batch_size"] == 32
cfg = json.load(open(os.path.join(ckpt, "config.json"), encoding="utf-8"))
stats = json.load(open(stats_path, encoding="utf-8"))
embedded = cfg.get("embedded_spline_stats")
assert embedded and all(embedded.get(key) == value for key, value in stats.items())
assert embedded["stats_contract_version"] == 1
assert embedded["boundary_semantics"] == "production_event_index_k_exclusive"
assert embedded["action_layout"] == "robocasa12"
assert embedded["degree"] == 3 and embedded["grip_idx"] == 11
assert embedded["pose_dims"] == [5, 6, 7, 8, 9, 10]
assert embedded["pass_dims"] == [0, 1, 2, 3, 4]
assert cfg["action_layout"] == "robocasa12" and cfg["predict_duration"] is True
assert (cfg["n_ctrl"], cfg["min_seg"], cfg["horizon_max"]) == (8, 8, 24)
with safe_open(os.path.join(ckpt, "model.safetensors"), framework="numpy") as tensors:
    for name in ("_tgt_mean", "_tgt_std"):
        value = tensors.get_tensor(name)
        assert value.shape == (8, 13) and np.isfinite(value).all(), name
    assert (tensors.get_tensor("_tgt_std") > 0).all()
print(json.dumps({"checkpoint": os.path.realpath(ckpt), "step": state["step"], "embedded_contract_exact": True}, sort_keys=True))
PY

mkdir -p "$MANIFEST_DIR"
set -o noclobber
: > "$CLAIM"
set +o noclobber

# n=10 is a predeclared triage sample. A nonzero or late-stage progress signal
# triggers a separately registered n>=30 distributional run; zero never proves
# impossibility.
KETTLE_JOB=$(sbatch --parsable --nice=10000 "$EVAL_LAUNCHER" "$CKPT" KettleBoiling 10 1500 "corr${ARM}100OSM_env" env 1000)
RINSE_JOB=$(sbatch --parsable --nice=10000 "$EVAL_LAUNCHER" "$CKPT" RinseSinkBasin 10 1350 "corr${ARM}100OSM_env" env 1000)
KETTLE_FIXED_JOB=""
KETTLE_ORACLE_JOB=""
if [[ "$ARM" == gran ]]; then
  # The task-level forensic video executes "turn burner on" before acquiring
  # the kettle. These two predeclared arms test whether ordered clause delivery
  # fixes that prerequisite failure. fixedk:227 is the source-demo median first
  # clause boundary; oracle is a mechanism upper bound and never a deployable
  # headline result.
  KETTLE_FIXED_JOB=$(sbatch --parsable --nice=10000 "$EVAL_LAUNCHER" "$CKPT" KettleBoiling 10 1500 "corr${ARM}100OSM_fixed227" fixedk:227 1000)
  KETTLE_ORACLE_JOB=$(sbatch --parsable --nice=10000 "$EVAL_LAUNCHER" "$CKPT" KettleBoiling 10 1500 "corr${ARM}100OSM_oracle" oracle 1000)
fi

python - "$RESULT" "$ARM" "$TRAIN_JOB" "$CKPT" "$KETTLE_JOB" "$RINSE_JOB" "$KETTLE_FIXED_JOB" "$KETTLE_ORACLE_JOB" <<'PY'
import hashlib, json, os, sys
path, arm, train_job, ckpt, kettle, rinse, kettle_fixed, kettle_oracle = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()
payload = {
    "schema": "robocasa_corrected_final_eval_gate_v1",
    "gate_job_id": os.environ["SLURM_JOB_ID"], "arm": arm,
    "training_terminal_job_id": train_job, "checkpoint": os.path.realpath(ckpt),
    "model_sha256": sha(os.path.join(ckpt, "model.safetensors")),
    "source_config_sha256": sha(os.path.join(ckpt, "config.json")),
    "contract_validated": True, "renderer": "osmesa", "n_per_task": 10,
    "diagnostic_only_until_distributional_replication": True,
    "eval_jobs": {
        "KettleBoiling_env": kettle,
        "RinseSinkBasin_env": rinse,
        "KettleBoiling_fixedk227": kettle_fixed or None,
        "KettleBoiling_oracle": kettle_oracle or None,
    },
}
with open(path, "x", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True); f.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "validated corrected ${ARM} 100k; submitted Kettle=${KETTLE_JOB} Rinse=${RINSE_JOB} fixed=${KETTLE_FIXED_JOB:-none} oracle=${KETTLE_ORACLE_JOB:-none}"
