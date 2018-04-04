#!/bin/bash
# Update documentation on gh-pages branch.

# Exit on errors
# set -o errexit -o nounset

# Ensure we are on master
if [ "$TRAVIS_BRANCH" = "master" ]; then

    # Make the docs
    pip install sphinx sphinx_rtd_theme
    cd $TRAVIS_BUILD_DIR/docs
    make html

    # Begin
    branch=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
    echo "Building docs from ${branch} branch..."

    # Get git hash
    rev=$(git rev-parse --short HEAD)

    # Copy the html folder to a temporary location, initialize
    # a new git repo there, add the necessary files, and force-push
    # to starry/gh-pages
    cd $TRAVIS_BUILD_DIR/docs/_build
    cp -r html tmp_html
    cd tmp_html
    git init
    touch .nojekyll
    git add -f .nojekyll
    git add -f *.html
    git add -f *.js
    git add -f _sources
    git add -f _static
    #git add -f _images
    git -c user.name='sphinx' -c user.email='sphinx' commit -m "rebuild gh-pages at ${rev}"
    git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG HEAD:gh-pages

    # Remove the temporary directory
    cd ..
    rm -rf tmp_html

fi
