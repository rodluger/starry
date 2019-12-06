#!/bin/bash
set -e

# Load the environment
if [[ -n $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Install dependencies
conda install -y -q -c conda-forge pandoc
pip install -U sphinx
pip install -U "nbconvert>=5.5"
pip install nbsphinx==0.5.0
pip install -U coverage-badge
pip install -U ipywidgets

# Get the current starry version
STARRY_VERSION=$(python .ci/print_starry_version.py)

# Build the docs
make -C docs html
cd docs/_build

# Clone the `gh-pages` branch
mkdir gh-pages
cd gh-pages
git clone -b gh-pages --single-branch https://github.com/rodluger/starry.git .

# Reset git tracking & update the current version's docs
rm -rf .git
rm -rf $STARRY_VERSION/*
mkdir -p $STARRY_VERSION
mv ../html/* $STARRY_VERSION/

# Copy over the coverage results
mkdir $STARRY_VERSION/coverage
git clone -b coverage --single-branch https://github.com/rodluger/starry.git $STARRY_VERSION/coverage
rm -rf $STARRY_VERSION/coverage/.git
mkdir $STARRY_VERSION/coverage-extensions
git clone -b coverage-extensions --single-branch https://github.com/rodluger/starry.git $STARRY_VERSION/coverage-extensions
rm -rf $STARRY_VERSION/coverage-extensions/.git

# Update the version list, if necessary
cp ../../../.ci/update_version_list.py .
python update_version_list.py $STARRY_VERSION

# Commit & force push back
git init
touch .nojekyll
git add .nojekyll
git add -f *
git -c user.name='rodluger' -c user.email='rodluger@gmail.com' \
    commit -m "rebuild gh-pages for ${STARRY_VERSION}"
git push -f https://$GHUSER:$GHKEY@github.com/rodluger/starry \
    HEAD:gh-pages >/dev/null 2>&1 -q
