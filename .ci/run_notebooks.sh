#!/bin/bash

# Load the environment
if [[ -e $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Run the notebooks using nbconvert
python .ci/run_notebooks.py

# Force push to `notebooks` branch
cd docs/notebooks
git init
git add -f *.ipynb
git -c user.name='rodluger' -c user.email='rodluger@gmail.com' \
    commit -m "rebuild notebooks"
git push -f https://$GHUSER:$GHKEY@github.com/rodluger/starry \
    HEAD:notebooks # DEBUG >/dev/null 2>&1 -q
