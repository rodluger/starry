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
sudo apt-get install ffmpeg

# Run tests
py.test -v -s tests/greedy --junitxml=junit/test-results-greedy.xml \
        --cov=starry --cov-append --cov-report html:coverage \
        --cov-config=.ci/.coveragerc \
        tests/greedy
py.test -v -s tests/lazy --junitxml=junit/test-results-lazy.xml --cov=starry \
         --cov-append --cov-report html:coverage \
         --cov-config=.ci/.coveragerc \
         tests/lazy

# Publish coverage results
if [[ -n $BUILDREASON ]] && [[ $BUILDREASON != "PullRequest" ]]; then
    coverage-badge -o coverage/coverage.svg
    cd coverage
    git init
    touch .nojekyll
    git add -f .nojekyll
    git add -f *
    git -c user.name='rodluger' -c user.email='rodluger@gmail.com' \
        commit -m "publish coverage"
    git push -f https://$GHUSER:$GHKEY@github.com/rodluger/starry \
        HEAD:coverage >/dev/null 2>&1 -q
fi
