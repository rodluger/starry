#!/bin/bash -x
#set -e

# Are there changes in the tex directory?
#if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'tex/'
#then

    #!/bin/bash -x
    cd $TRAVIS_BUILD_DIR/tex/
    python falinks.py
    tectonic starry.tex --print

    # Force push the paper to GitHub
    cd $TRAVIS_BUILD_DIR
    git checkout --orphan $TRAVIS_BRANCH-pdf
    git rm -rf . > /dev/null 2>&1
    git add -f tex/starry.pdf
    git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
    git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf

#fi
