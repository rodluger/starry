# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import sys
import os

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__STARRY_DOCS__ = True
import starry

# Copy notebooks over
import copy_notebooks
import glob

if not os.path.exists("notebooks"):
    os.mkdir("notebooks")
for infile in glob.glob("../notebooks/*.ipynb"):
    outfile = os.path.join("notebooks", os.path.basename(infile))
    copy_notebooks.copy(infile, outfile)

# -- Project information -----------------------------------------------------

project = "starry"
copyright = "2019, Rodrigo Luger"
author = "Rodrigo Luger"

# The full version, including alpha/beta/rc tags
version = starry.__version__
release = starry.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {"display_version": True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]

html_js_files = ["js/version.js"]
html_css_files = ["css/hide_input.css"]

# -- Extension settings ------------------------------------------------------

from ipywidgets.embed import DEFAULT_EMBED_REQUIREJS_URL

html_js_files += [DEFAULT_EMBED_REQUIREJS_URL]

# Add a heading to notebooks
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. note:: This tutorial was generated from a Jupyter notebook that can be
          downloaded `here <https://github.com/rodluger/starry/blob/dev/docs/{{ docname }}>`_.
"""

# Remove jupyter notebook prompt numbers
nbsphinx_prompt_width = 0
napoleon_use_ivar = True

# Todo lists
todo_include_todos = True

autosummary_generate = True

autodoc_docstring_signature = True
