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

# Run the notebooks using nbconvert
mkdir -p docs/notebooks
python .ci/run_notebooks.py

# Force push to `notebooks` branch
if [[ -n $BUILDREASON ]] && [[ $BUILDREASON != "PullRequest" ]]; then
    cd docs/notebooks
    git init
    git add -f *.ipynb
    git add -f *.log
    git -c user.name='rodluger' -c user.email='rodluger@gmail.com' \
        commit -m "rebuild notebooks"
    git push -f https://$GHUSER:$GHKEY@github.com/rodluger/starry \
        HEAD:notebooks >/dev/null 2>&1 -q
fi
