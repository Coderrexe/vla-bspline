#!/bin/bash
#SBATCH --job-name=fix_libero
#SBATCH --partition=day
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
# Fix the LIBERO sim extra: egl_probe fails because (1) EGL/GL dev headers are missing,
# (2) CMake 4.x rejects egl_probe's old CMakeLists. Provide headers via conda-forge mesalib
# and set CMAKE_POLICY_VERSION_MINIMUM=3.5 so CMake 4.x will configure the old project.

set -uo pipefail
module load miniconda
ENV=$HOME/.conda/envs/lerobot
source activate $ENV
cd $HOME/vla_bspline/lerobot

# EGL/GL/GLES headers + libs (installs $CONDA_PREFIX/include/{EGL,GL,GLES*})
conda install -y -c conda-forge mesalib libglu

echo "=== EGL header now present? ==="
ls $CONDA_PREFIX/include/EGL/egl.h $CONDA_PREFIX/include/GL/gl.h 2>&1

export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/include
export LIBRARY_PATH=$CONDA_PREFIX/lib

# build egl_probe alone first (fast to see if the fix works), then the full libero extra
pip install --no-cache-dir --no-build-isolation egl_probe 2>&1 | tail -20
echo "=== egl_probe exit: $? ==="
pip install -e ".[libero]" 2>&1 | tail -25
echo "=== [libero] exit: $? ==="

echo "=== VERIFY libero import ==="
python -c "import libero; print('libero OK', getattr(libero,'__file__','?'))" 2>&1
python -c "import egl_probe; print('egl_probe OK')" 2>&1
