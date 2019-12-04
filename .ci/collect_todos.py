import json
import os
import numpy as np


snippet_length_before = 4
snippet_length_after = 8

# Base url of the commit on GitHub
url = "https://github.com/rodluger/starry/blob/{}".format(
    os.getenv("GHSHA1", "master")
)

template = """
`{} <{}/{}#L{}>`_:

.. code-block:: {}
    :lineno-start: {}
    :emphasize-lines: {}

    {}

"""

# Get the JSON data from leasot
with open("docs/todos.json", "r") as f:
    data = json.load(f)

if len(data) == 0:
    verbs = "are", "need"
    value = "no more"
elif len(data) == 1:
    verb = "is", "needs"
    value = "one"
else:
    verb = "are", "need"
    value = len(data)

contents = """
todos
-----

There {} **{}** todo items that {} to be addressed in ``starry``.

""".format(
    verbs[0], value, verbs[1]
)

# Parse each entry and add to the .rst file
for entry in data:

    file = entry["file"]
    line = entry["line"]
    text = entry["text"]
    if file.endswith(".py"):
        language = "python"
    elif file.endswith(".cpp") or file.endswith(".h"):
        language = "c++"
    else:
        language = ""

    # Get a snippet around the line
    with open(file, "r") as f:
        lines = f.readlines()
    inds = (
        np.arange(-snippet_length_before, snippet_length_after, dtype=int)
        + line
        - 1
    )
    inds = inds[inds > 0]
    inds = inds[inds < len(lines)]
    snippet = "    ".join([lines[i] for i in inds])

    # Add to file
    contents += template.format(
        os.path.basename(file),
        url,
        file,
        line,
        language,
        inds[0] + 1,
        line - inds[0],
        snippet,
    )

# Write to file
with open("docs/todos.rst", "w") as f:
    print(contents, file=f)
