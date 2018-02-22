#!/bin/bash -x
set -e

if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'tex/'
then
    # Conda env
    #conda create --yes -n paper
    #source activate paper
    conda install -c conda-forge -c pkgw-forge tectonic

    # Generate the figures
    cd tex/figures
    for f in *.py; do
        python "$f"
    done

    # Build the paper using tectonic
    cd ../
    tectonic starry.tex --print

    # Force push the paper to GitHub
    cd $TRAVIS_BUILD_DIR
    git checkout --orphan pdf
    git rm -rf .
    git add -f tex/starry.pdf
    git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
    git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG pdf
fi
