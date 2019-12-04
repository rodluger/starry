#!/bin/bash
#set -e

# Load the environment
if [[ -n $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Install leasot
npm install --global leasot

# Grab the todos
leasot -S -i starry/lib/vendor starry tests -r json > docs/todos.json

# Generate the .rst file
python .ci/collect_todos.py
