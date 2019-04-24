"""
Various hacks to beautify the docs on travis.

"""

import starry
import re
import os

# Update the current version number in Doxygen
with open("Doxyfile", "r") as f:
    file = f.read()
file = re.sub('PROJECT_NUMBER\s*?= "(.*?)"\n', 
              'PROJECT_NUMBER = "%s"\n' % starry.__version__, file)
with open("Doxyfile", "w") as f:
    print(file, file=f)

# Update the commit and branch in the Sphinx footer
commit = os.getenv("TRAVIS_COMMIT", "unknown")
commit_url = "https://github.com/rodluger/starry/tree/%s" % commit
branch_name = os.getenv("TRAVIS_BRANCH", "unknown")
branch_url = "https://github.com/rodluger/starry/tree/%s" % branch_name

with open("sphinx_rtd_theme/static/js/theme.js", "r") as f:
    file = f.read()
file = re.sub('var commit_url = "(.*?)";\n', 
              'var commit_url = "%s";\n' % commit_url, file)
file = re.sub('var branch_url = "(.*?)";\n', 
              'var branch_url = "%s";\n' % branch_url, file)
file = re.sub('var branch_name = "(.*?)";\n', 
              'var branch_name = "%s";\n' % branch_name, file)
with open("sphinx_rtd_theme/static/js/theme.js", "w") as f:
    print(file, file=f)