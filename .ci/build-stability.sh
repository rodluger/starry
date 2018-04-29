#!/bin/bash -x
set -e

# Run stability tests
cd $TRAVIS_BUILD_DIR/tests
python stability.py

# Force push to the stability branch
cd $TRAVIS_BUILD_DIR/tests
git init
git add -f *.pdf
git -c user.name='travis' -c user.email='travis' commit -m "running stability tests"
git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG HEAD:stability-pdf

# Return to the top level
cd $TRAVIS_BUILD_DIR
