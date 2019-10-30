#!/bin/bash

# Load the environment
if [[ -e $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Clone the `notebooks` branch
git clone -b notebooks --single-branch https://github.com/rodluger/starry.git docs/notebooks
