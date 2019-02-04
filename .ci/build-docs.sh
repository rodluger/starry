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

# Initialize a new git repo in the build dir,
# add the necessary files, and force-push
# to the gh-pages branch
cd $TRAVIS_BUILD_DIR/docs/.build/html
git init
touch .nojekyll
git add -f .nojekyll
git add -f *

# DEBUG
doxygen -g
git add Doxyfile

git -c user.name='sphinx' -c user.email='sphinx' commit -m "rebuild gh-pages at ${rev}"
git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/rodluger/starry2 HEAD:gh-pages

# Return to the top level
cd $TRAVIS_BUILD_DIR
