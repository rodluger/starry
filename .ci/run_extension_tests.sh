#!/bin/bash

# Load the environment
if [[ -e $CONDA ]]; then
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

# Publish coverage results
if [ $BUILDREASON != "PullRequest" ]; then
    coverage-badge -n extensions.svg -o coverage-extensions/coverage.svg
    cd coverage-extensions
    git init
    touch .nojekyll
    git add -f .nojekyll
    git add -f *
    git -c user.name='rodluger' -c user.email='rodluger@gmail.com' \
        commit -m "publish coverage"
    git push -f https://$GHUSER:$GHKEY@github.com/rodluger/starry \
        HEAD:coverage-extensions >/dev/null 2>&1 -q
fi
