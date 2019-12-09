#!/bin/bash

# Load the environment
if [[ -n $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Install dependencies
pip install -U parameterized nose pytest pytest-cov
pip install -U starry_beta
pip install -U git+https://github.com/rodluger/coverage-badge

# Run tests
py.test -v -s starry/extensions/tests/greedy \
        --junitxml=junit/test-results-extensions-greedy.xml --cov=starry/extensions \
        --cov-append --cov-report html:coverage-extensions \
        --cov-config=.ci/.coveragerc_extensions \
        starry/extensions/tests/greedy
py.test -v -s starry/extensions/tests/lazy \
        --junitxml=junit/test-results-extensions-lazy.xml --cov=starry/extensions \
        --cov-append --cov-report html:coverage-extensions \
        --cov-config=.ci/.coveragerc_extensions \
        starry/extensions/tests/lazy

# Generate badge
coverage-badge -n extensions.svg -o coverage-extensions/coverage.svg
