#!/bin/bash
set -e

# Display machine info
lscpu

# Set up conda
sudo chown -R $USER $CONDA
. $CONDA/etc/profile.d/conda.sh
conda create --yes --quiet --name starry python=3.7.3 pip

# Activate conda & install base dependencies
. $CONDA/etc/profile.d/conda.sh
conda activate starry
conda install -y -q numpy scipy mkl openblas=0.3.6 theano
theano-cache purge
pip install -U pip
pip install -U setuptools
pip install -U -r requirements.txt
pip install -U -r optional-requirements.txt
pip install setuptools_scm

# Install starry
python setup.py develop
