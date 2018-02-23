#!/bin/bash -x
set -e

if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'tex/'
then

    # Build the paper
    cd tex/
	pdflatex -interaction=nonstopmode -halt-on-error starry.tex
	bibtex starry
	( grep Rerun starry.log && pdflatex -interaction=nonstopmode -halt-on-error starry.tex ) || echo "Done."
	( grep Rerun starry.log && pdflatex -interaction=nonstopmode -halt-on-error starry.tex ) || echo "Done."

    # Force push the paper to GitHub
    cd $TRAVIS_BUILD_DIR
    git checkout --orphan pdf
    git rm -rf .
    git add -f tex/starry.pdf
    git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
    git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG pdf

fi
