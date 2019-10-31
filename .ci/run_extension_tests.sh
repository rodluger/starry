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
        --cov-append --cov-report html:coverage-extensions \
        --cov-config=.coveragerc_extensions \
        starry/extensions/tests/greedy
py.test -v -s starry/extensions/tests/lazy \
        --junitxml=junit/test-results-extensions-lazy.xml --cov=starry \
        --cov-append --cov-report html:coverage-extensions \
        --cov-config=.coveragerc_extensions \
        starry/extensions/tests/lazy

# Get coverage badge (hackily edit the label)
coverage-badge -o tmp.svg
python -c "f1 = open('tmp.svg', 'r'); f2 = open('coverage-extensions/coverage.svg', 'w'); [f2.write(l.replace('coverage', 'extensions coverage')) for l in f1.readlines()]"

# Force push to `coverage-extensions` branch
cd coverage-extensions
git init
touch .nojekyll
git add -f .nojekyll
git add -f *
git -c user.name='rodluger' -c user.email='rodluger@gmail.com' \
    commit -m "publish coverage"
git push -f https://$GHUSER:$GHKEY@github.com/rodluger/starry \
    HEAD:coverage-extensions >/dev/null 2>&1 -q
