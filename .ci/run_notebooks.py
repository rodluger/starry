import glob
import json
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def warning_filter(infile, outfile):
    logfile = os.path.join(
        os.path.dirname(os.path.abspath(outfile)), "warnings.log"
    )
    header = os.path.basename(infile)
    div = "=" * len(header)
    header = "\n\n" + div + "\n" + header + "\n" + div + "\n\n"
    return """import warnings, sys

def customwarn(message, category, filename, lineno, file=None, line=None):
    with open("{logfile}", "a") as file:
        print(
            warnings.formatwarning(message, category, filename, lineno),
            file=file,
        )

with open("{logfile}", "a") as file:
    print('''{header}''', file=file)

warnings.showwarning = customwarn""".format(
        logfile=logfile, header=header
    )


matplotlib_setup = """get_ipython().magic('config InlineBackend.figure_format = "retina"')
import matplotlib.pyplot as plt
plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["font.cursive"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "custom"
"""


def run(infile, outfile, timeout=1200):
    print("Executing %s..." % infile)

    # Open the notebook
    with open(infile, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Filter warnings
    nb.get("cells").insert(
        0,
        nbformat.notebooknode.NotebookNode(
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {"tags": ["hide_input"]},
                "outputs": [],
                "source": warning_filter(infile, outfile),
            }
        ),
    )

    # Matplotlib setup
    nb.get("cells").insert(
        0,
        nbformat.notebooknode.NotebookNode(
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {"tags": ["hide_input"]},
                "outputs": [],
                "source": matplotlib_setup,
            }
        ),
    )

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
    # Run the notebookss
    files = glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb"))
    for infile in files:
        outfile = os.path.join(
            ROOT, "docs", "notebooks", os.path.basename(infile)
        )
        run(infile, outfile)
