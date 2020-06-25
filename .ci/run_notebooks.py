import glob
import json
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(infile, outfile, timeout=2400):
    print("Executing %s..." % infile)

    # Open the notebook
    with open(infile, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    if nb["metadata"].get("nbsphinx_execute", True):
        try:
            ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
            ep.preprocess(
                nb,
                {
                    "metadata": {
                        "path": os.path.dirname(os.path.abspath(infile))
                    }
                },
            )
        except Exception as error:
            if "[allow-failures]" in os.getenv("COMMIT", ""):
                pass
            else:
                raise error

    # Replace input in certain cells
    for cell in nb.get("cells", []):
        replace_input_with = cell["metadata"].get("replace_input_with", None)
        if replace_input_with is not None:
            cell["source"] = replace_input_with

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
