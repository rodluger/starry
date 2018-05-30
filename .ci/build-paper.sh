#!/bin/bash -x
#set -e

# Are there changes in the tex directory?
if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'tex/'
then

    # Install texlive
    sudo apt-get -qq update && sudo apt-get install -y --no-install-recommends texlive-full

    # Generate the figures
    echo "Generating figures..."
    cd tex/figures
    for f in *.py; do
        echo "Running $f..."
        python "$f"
    done
    cd ../../

    # Build the paper
    cd tex/
	pdflatex -interaction=nonstopmode -halt-on-error starry.tex
	bibtex starry
	pdflatex -interaction=nonstopmode -halt-on-error starry.tex
	pdflatex -interaction=nonstopmode -halt-on-error starry.tex
    pdflatex -interaction=nonstopmode -halt-on-error starry.tex

    # Force push the paper to GitHub
    cd $TRAVIS_BUILD_DIR
    git checkout --orphan $TRAVIS_BRANCH-pdf
    git rm -rf . > /dev/null 2>&1
    git add -f tex/starry.pdf
    git add -f tex/stability_test*.pdf
    git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
    git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf

fi
