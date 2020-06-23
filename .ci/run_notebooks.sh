#!/bin/bash
set -e

# Load the environment
if [[ -n $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Attempt to fix BLAS warnings
sudo apt-get update
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
export THEANO_FLAGS=blas.ldflags="-L/usr/lib/ -lblas"

# Install dependencies
pip install -U "nbconvert>=5.5"
pip install -U ipywidgets
sudo apt-get install ffmpeg
pip install -U sympy
pip install -U corner
pip install -U batman-package
pip install -U tqdm

# Run the notebooks using nbconvert
mkdir -p docs/notebooks
python .ci/run_notebooks.py
