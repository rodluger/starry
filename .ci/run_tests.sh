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
        --cov=starry --cov-append --cov-report html:htmlcov tests/greedy
py.test -v -s tests/lazy --junitxml=junit/test-results-lazy.xml --cov=starry \
         --cov-append --cov-report html:htmlcov tests/lazy

# Get coverage badge
coverage-badge -o htmlcov/coverage.svg
