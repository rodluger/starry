import glob
import json
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(infile, outfile, timeout=1200):
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
    if nb["metadata"].get("nbsphinx_execute", True):
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        ep.preprocess(
            nb,
            {"metadata": {"path": os.path.dirname(os.path.abspath(infile))}},
        )

    # Write it back
    with open(outfile, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    # Run the notebookss
    files = glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb"))
    for infile in files:
        outfile = os.path.join(
            ROOT, "docs", "notebooks", os.path.basename(infile)
        )
        run(infile, outfile)
