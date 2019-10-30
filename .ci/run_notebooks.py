import glob
import json
import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run(infile, outfile, timeout=600):
    print("Executing %s..." % infile)

    # Open the notebook
    with open(infile, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Process custom tags
    for cell in nb.get("cells", []):
        if "hide_input" in cell.get("metadata", {}).get("tags", []):
            cell["source"] = "#hide_input\n" + cell["source"]
        if "hide_output" in cell.get("metadata", {}).get("tags", []):
            cell["source"] = "#hide_output\n" + cell["source"]

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(
        nb, {"metadata": {"path": os.path.dirname(os.path.abspath(infile))}}
    )

    # Write it back
    with open(outfile, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    for infile in glob.glob("../notebooks/*.ipynb"):
        outfile = os.path.join("../docs/notebooks", os.path.basename(infile))
        run(infile, outfile)
