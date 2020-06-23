#!/bin/bash

# Load the environment
if [[ -n $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Install dependencies
pip install -U parameterized nose pytest pytest-cov
pip install -U starry_beta
pip install -U tqdm
pip install -U git+https://github.com/rodluger/coverage-badge

# Run tests
py.test -v -s tests/greedy --junitxml=junit/test-results-greedy.xml \
        --cov=starry --cov-append --cov-report html:coverage \
        --cov-config=.ci/.coveragerc \
        tests/greedy
py.test -v -s tests/lazy --junitxml=junit/test-results-lazy.xml --cov=starry \
         --cov-append --cov-report html:coverage \
         --cov-config=.ci/.coveragerc \
         tests/lazy

# Generate badge
coverage-badge -o coverage/coverage.svg
