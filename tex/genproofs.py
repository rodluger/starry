import re
import numpy as np

header = """Proofs
======

Below you'll find proofs, derivations, or numerical validations
of the principal equations in the :py:obj:`starry` paper.

.. toctree::
   :titlesonly:
   :maxdepth: 1

"""

# Open the LaTeX log
try:
    with open("../tex/starry.log", "r") as f:
        log = f.read()
except FileNotFoundError:
    # TeX file not compiled. Let's not change anything.
    exit()

# Grab the equation labels and tags
pairs = re.findall("<<<(.*?)\: ([0-9]*?)>>>", log)
pairs += re.findall("<<<(.*?)\: \\\\hbox \{(.*?)\}>>>", log)
labels = []
tags = []
for pair in pairs:
    if not pair[0] in labels:
        labels.append(pair[0])
        tags.append([pair[1]])
    else:
        idx = np.argmax([label == pair[0] for label in labels])
        tags[idx].append(pair[1])

# Print these to the table of contents in `proofs.rst`
with open("../docs/proofs.rst", "w") as f:
    print(header, file=f)
    for i, _ in enumerate(tags):
        if len(tags[i]) == 1:
            text = "Equation (%s)" % tags[i][0]
        else:
            text = "Equations %s" % ", ".join(["(" + tag + ")"
                                               for tag in tags[i]])
        line = "   %s <proofs/%s.ipynb>" % (text, labels[i])
        print(line, file=f)
