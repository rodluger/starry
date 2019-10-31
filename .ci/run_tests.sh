#!/bin/bash

# Load the environment
if [[ -e $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Install dependencies
pip install -U parameterized nose pytest pytest-cov coverage-badge
pip install -U starry_beta

# Run tests
py.test -v -s tests/greedy --junitxml=junit/test-results-greedy.xml \
        --cov=starry --cov-append --cov-report html:coverage \
        --cov-config=.coveragerc \
        tests/greedy
py.test -v -s tests/lazy --junitxml=junit/test-results-lazy.xml --cov=starry \
         --cov-append --cov-report html:coverage \
         --cov-config=.coveragerc \
         tests/lazy

# Get coverage badge
coverage-badge -o coverage/coverage.svg

# Force push to `coverage` branch
cd coverage
git init
touch .nojekyll
git add -f .nojekyll
git add -f *
git -c user.name='rodluger' -c user.email='rodluger@gmail.com' \
    commit -m "publish coverage"
git push -f https://$GHUSER:$GHKEY@github.com/rodluger/starry \
    HEAD:coverage >/dev/null 2>&1 -q
