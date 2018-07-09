#!/bin/bash -x
set -e

# Are there changes in the tex directory?
if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'tex/'
then

    # Build the paper
    echo "Building the paper..."
    cd $TRAVIS_BUILD_DIR/tex && make

    # If `proofs.rst` changed, let's commit it
    if git diff --name-only | grep 'proofs.rst'
    then
        git add $TRAVIS_BUILD_DIR/docs/proofs.rst
        git commit -m "updating proofs.rst [skip ci]"
        git push -q https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH
    fi

    # Force push the paper to GitHub
    mkdir -p travis && cd travis
    mkdir -p tex && mv ../starry.pdf tex/
    git init
    git add -f tex/starry.pdf
    git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
    git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf
    cd $TRAVIS_BUILD_DIR

fi
