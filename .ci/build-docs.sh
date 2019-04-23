#!/bin/bash
set -e

# Make the docs
cd $TRAVIS_BUILD_DIR/docs
doxygen -v
make html

# Begin
branch=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
echo "Building docs from ${branch} branch..."

# Get git hash
rev=$(git rev-parse --short HEAD)

# Get the current starry version
STARRY_VERSION=$(python -c "import starry; print('v' + starry.__version__)")

# Go to the html build
cd $TRAVIS_BUILD_DIR/docs/_build/

# Clone the `gh-pages` branch
mkdir gh-pages
cd gh-pages
git clone -b gh-pages --single-branch https://github.com/rodluger/starry.git .

# Reset git tracking & update the current version's docs
rm -rf .git
mv ../html $STARRY_VERSION

# Commit & force push back
git init
touch .nojekyll
git add -f .nojekyll
git add -f *
git -c user.name='sphinx' -c user.email='sphinx' commit -m "rebuild gh-pages at ${rev} for ${STARRY_VERSION}"
git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG HEAD:gh-pages

# Return to the top level
cd $TRAVIS_BUILD_DIR
