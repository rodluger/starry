#!/bin/bash
set -e

# Load the environment
if [[ -e $CONDA ]]; then
    . $CONDA/etc/profile.d/conda.sh
    conda activate starry
fi

# Install dependencies
conda install -y -q -c conda-forge pandoc
pip install -U sphinx
pip install -U "nbconvert>=5.5"
git clone https://github.com/rodluger/nbsphinx
pushd nbsphinx
python setup.py develop
popd
pip install -U coverage-badge
pip install -U ipywidgets

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
git push -f https://$GHUSER:$GHKEY@github.com/rodluger/starry \
    HEAD:gh-pages # DEBUG >/dev/null 2>&1 -q
