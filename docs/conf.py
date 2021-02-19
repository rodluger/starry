# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import starry
import sys

# Add the CWD to the path
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

# Custom hacks for this project
import hacks


# -- Project information -----------------------------------------------------

project = "starry"
copyright = "2021, Rodrigo Luger"
author = "Rodrigo Luger"
version = starry.__version__
release = version

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
html_last_updated_fmt = "%Y %b %d at %H:%M:%S UTC"
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]

# -- Extension settings ------------------------------------------------------

# autodocs
autoclass_content = "both"
autosummary_generate = True
autodoc_docstring_signature = True

# Add a heading to notebooks
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. note:: This tutorial was generated from a Jupyter notebook that can be
          downloaded `here <https://github.com/rodluger/starry/blob/master/{{ docname }}>`_.
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

# nbsphinx
nbsphinx_timeout = 600
napoleon_use_ivar = True

# todos
todo_include_todos = True
