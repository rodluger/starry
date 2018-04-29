#!/bin/bash -x
set -e

# Run stability tests
cd $TRAVIS_BUILD_DIR/tests
python stability.py

# Force push to the stability branch
cd $TRAVIS_BUILD_DIR
git checkout --orphan stability-pdf
git rm -rf . > /dev/null 2>&1
git add -f tests/*.pdf
git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG stability-pdf
