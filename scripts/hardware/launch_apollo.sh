#!/usr/bin/env bash
# Lab-only launcher. Defaults to prediction-only SHADOW mode in the Python CLI.
# Never starts a runtime/session, homes an arm, or changes the lab environment.
set -euo pipefail
APOLLO_VLA_ROOT=/home/mavis-v2/simba/vla_bspline_hardware_20260912
test -x "$APOLLO_VLA_ROOT/.venv/bin/python"
test -d "$APOLLO_VLA_ROOT/vendor/policy-node/mavis_policy_node"
cd "$APOLLO_VLA_ROOT"
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$APOLLO_VLA_ROOT/bundle/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH="$APOLLO_VLA_ROOT/code:$APOLLO_VLA_ROOT/source/src:$APOLLO_VLA_ROOT/vendor/policy-node"
exec "$APOLLO_VLA_ROOT/.venv/bin/python" -u "$APOLLO_VLA_ROOT/code/run_apollo_dora.py" "$@"
