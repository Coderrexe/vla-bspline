#!/bin/bash
#SBATCH --job-name=fix_libero2
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
# Install hf-egl-probe with --no-build-isolation (the trick that made plain egl_probe build),
# so its setup.py sees the env's cmake + EGL headers instead of an isolated build env.
set -uo pipefail
module load miniconda
ENV=$HOME/.conda/envs/lerobot
source activate $ENV
cd $HOME/vla_bspline/lerobot

export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/include
export LIBRARY_PATH=$CONDA_PREFIX/lib
export PATH=$CONDA_PREFIX/bin:$PATH

echo "=== which cmake: $(which cmake) | $(cmake --version | head -1) ==="
# build deps present in-env so no isolation needed
pip install --no-build-isolation "hf-egl-probe" 2>&1 | tail -15
echo "=== hf-egl-probe exit: $? ==="
pip install -e ".[libero]" 2>&1 | tail -15
echo "=== [libero] exit: $? ==="
python -c "import libero; print('libero OK', libero.__file__)" 2>&1
python -c "import hf_egl_probe" 2>&1 || python -c "import egl_probe; print('egl_probe OK')" 2>&1
