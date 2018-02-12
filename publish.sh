#!/bin/bash

# Are we on travis?
if [ -n "$GITHUB_API_KEY" ]; then
  cd $TRAVIS_BUILD_DIR
  cd tex
  git checkout --orphan pdf
  git rm -rf .
  git add -f starry.pdf
  git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
  git push -q -f https://rodluger:$GITHUB_API_KEY@github.com/rodluger/starry pdf
else
  # Create a temporary directory and copy the pdf over
  rm -rf ../.starry-pdf
  mkdir ../.starry-pdf
  cp tex/starry.pdf ../.starry-pdf
  cd ../.starry-pdf
  # Initialize a git repo and force-push to the pdf branch
  git init
  git checkout --orphan pdf
  git add -f starry.pdf
  git -c user.name='pdf' -c user.email='pdf' commit -m "building the paper"
  git push -q -f https://rodluger@github.com/rodluger/starry pdf
  # Back to where we were
  cd ../starry
  rm -rf ../.starry-pdf
fi
