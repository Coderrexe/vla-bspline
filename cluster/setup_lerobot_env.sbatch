#!/bin/bash
#SBATCH --job-name=lerobot_env
#SBATCH --partition=day
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
# Build a fresh lerobot env (SmolVLA + LIBERO) for the B-spline VLA project.
# Runs on a compute node (has internet) per YCRC etiquette — NOT the login node.
# Env goes in $HOME/.conda/envs (home has ~68G free; shared project/scratch are ~93% full).

set -uo pipefail
echo "=== node: $(hostname) | start: $(date) ==="

module load miniconda
export CONDA_PKGS_DIRS=$HOME/.conda/pkgs
ENV_PREFIX=$HOME/.conda/envs/lerobot
PROJ=$HOME/vla_bspline
mkdir -p "$PROJ"

# --- 1. create env (lerobot requires python >= 3.12) ---
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.12
fi
PY=$ENV_PREFIX/bin/python
PIP=$ENV_PREFIX/bin/pip
echo "=== python: $($PY -V) ==="
$PIP install --upgrade pip

# --- 1b. build toolchain for source deps (egl_probe/hf-egl-probe need CMake + a C++ compiler) ---
conda install -y -p "$ENV_PREFIX" -c conda-forge cmake ninja
module load GCC 2>/dev/null || true
echo "=== cmake: $($ENV_PREFIX/bin/cmake --version 2>&1 | head -1) | gcc: $(gcc --version 2>&1 | head -1) ==="

# --- 2. clone lerobot ---
cd "$PROJ"
if [ ! -d lerobot ]; then
  git clone --depth 1 https://github.com/huggingface/lerobot.git
fi
cd lerobot
echo "=== lerobot commit: $(git rev-parse HEAD) ==="

# --- 3. install in stages so a sim-dep build failure can't wipe the training env ---
# 3a. core training stack (smolvla policy + wandb/accelerate) — all wheels, should be robust
$PIP install -e ".[smolvla,training]"
echo "=== [smolvla,training] exit: $? ==="
# 3b. LIBERO sim env (pulls hf-libero -> egl_probe, which compiles) — isolated so failure is contained
$PIP install -e ".[libero]"
echo "=== [libero] exit: $? ==="

# --- 4. verify ---
echo "=== VERIFY ==="
$PY - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
for mod in ["lerobot", "torch", "transformers", "datasets"]:
    try:
        m = importlib.import_module(mod)
        print(f"OK  {mod:14s}", getattr(m, "__version__", "?"))
    except Exception as e:
        print(f"ERR {mod:14s}", repr(e)[:150])
import torch
print("torch cuda build:", torch.version.cuda, "| cuda avail (CPU node -> False expected):", torch.cuda.is_available())
# policy + env registration
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("OK  SmolVLAPolicy import")
except Exception as e:
    print("ERR SmolVLAPolicy", repr(e)[:200])
try:
    import libero, hf_libero  # noqa
    print("OK  hf-libero import")
except Exception as e:
    print("ERR hf-libero", repr(e)[:200])
PY
echo "=== done: $(date) ==="
