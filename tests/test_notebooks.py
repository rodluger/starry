"""Convert all the tutorials to python scripts and run them as tests."""
import glob
import os
from nbconvert import PythonExporter
import re

# Set the benchmark flag
__benchmark__ = True


def test_notebooks():
    """Run all notebooks in /docs/tutorials/ as tests."""
    # Get the notebook names
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'docs', 'tutorials')
    notebooks = glob.glob(os.path.join(path, '*.ipynb'))

    # Convert them to python scripts
    exporter = PythonExporter()
    for notebook in notebooks:
        # Get the script as a string
        script, _ = exporter.from_filename(notebook)

        # Get rid of %matplotlib inline commands
        script = script.replace("get_ipython().magic('matplotlib inline')", "")
        script = script.replace(
            "get_ipython().run_line_magic('matplotlib', 'inline')", "")

        # Remove the %time wrappers
        script = re.sub("get_ipython\(\).magic\('time (.*)'\)", r"\1", script)

        # DEBUG
        if "ipython" in script:
            print(script)
            continue

        # Run it
        print("Running %s..." % os.path.basename(notebook))
        exec(script, globals(), globals())


if __name__ == "__main__":
    test_notebooks()
