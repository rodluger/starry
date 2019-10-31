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
py.test -v -s starry/extensions/tests/greedy \
        --junitxml=junit/test-results-extensions-greedy.xml --cov=starry \
        --cov-append --cov-report html:htmlcov starry/extensions/tests/greedy
py.test -v -s starry/extensions/tests/lazy \
        --junitxml=junit/test-results-extensions-lazy.xml --cov=starry \
        --cov-append --cov-report html:htmlcov starry/extensions/tests/lazy

# Get coverage badge
coverage-badge -o htmlcov/coverage.svg
