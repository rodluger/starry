#!/bin/bash

# Load the environment
if [[ -e $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Build the docs
make -C docs html
cd docs/_build

# Clone the `gh-pages` branch
mkdir gh-pages
cd gh-pages
git clone -b gh-pages --single-branch https://github.com/rodluger/starry.git .

# Reset git tracking & update the current version's docs
rm -rf .git
STARRY_VERSION=$(python -c "import starry; print('v' + starry.__version__)")
rm -rf $STARRY_VERSION/*
mkdir -p $STARRY_VERSION
mv ../html/* $STARRY_VERSION/

# Copy over the coverage results, if available
mv ../../../htmlcov $STARRY_VERSION/

# Update the version list, if necessary
python update_version_list.py $STARRY_VERSION

# Commit & force push back
git init
touch .nojekyll
git add -f .nojekyll
git add -f *
git -c user.name='rodluger' -c user.email='rodluger@gmail.com' \
    commit -m "rebuild gh-pages for ${STARRY_VERSION}"
git push -f https://$(GITHUB_USER):$(GITHUB_API_KEY)@github.com/rodluger/starry \
    HEAD:gh-pages >/dev/null 2>&1 -q
