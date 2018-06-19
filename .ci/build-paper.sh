#!/bin/bash -x
set -e

# Are there changes in the tex directory?
if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'tex/'
then

    export PATH=/tmp/texlive/bin/x86_64-linux:$PATH
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    if ! command -v texlua > /dev/null; then
      # Obtain TeX Live
      wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
      tar -xzf install-tl-unx.tar.gz
      cd install-tl-20*

      # Install a minimal system
      ./install-tl --profile=$DIR/texlive.profile

      cd ..
    fi

    tlmgr install luatex

    tlmgr install \
      l3kernel \
      l3packages \
      listings \
      pgf \
      tools \
      graphics \
      xkeyval \
      hyperref \
      xcolor \
      cleveref \
      etoolbox \
      oberdiek \
      ifxetex \
      ifluatex \
      tools \
      url \
      parskip \
      xstring \
      fontspec \
      fontawesome \
      lipsum \
      zapfding \
      luaotfload \
      cjk \
      xecjk \
      fandol \
      dvipdfmx \
      microtype \
      url \
      amsmath \
      mathtools \
      esint \
      amsfonts \
      natbib \
      multirow \
      scalerel \
      etoolbox \
      marginnote \
      units \
      tabstackengine \
      diagbox \
      cancel \
      mathdots \
      bbm \
      booktabs \
      was \
      fontawesome \
      listings

    # Keep no backups (not required, simply makes cache bigger)
    tlmgr option -- autobackup 0

    # Update the TL install but add nothing new
    tlmgr update --self --all --no-auto-install

    # Generate the figures
    echo "Generating figures..."
    cd $TRAVIS_BUILD_DIR/tex/figures
    for f in *.py; do
        echo "Running $f..."
        python "$f"
    done
    cd ../../

    # Build the paper
    cd tex/
    python falinks.py
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
    git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
    git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf

fi
