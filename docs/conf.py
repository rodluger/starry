# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import starry
from ipywidgets.embed import DEFAULT_EMBED_REQUIREJS_URL
import sys

# -- HACK the modules for the docs --------------------------------------------

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__STARRY_DOCS__ = True


class CustomBase(object):
    @property
    def amp(self):
        """
        The overall amplitude of the map in arbitrary units. This factor
        multiplies the intensity and the flux and is thus proportional to the
        luminosity of the object. For multi-wavelength maps, this is a vector
        corresponding to the amplitude of each wavelength bin.
        """
        pass


class _Map(CustomBase, starry.maps.YlmBase, starry.maps.MapBase):
    __doc__ = starry.maps.YlmBase.__doc__


class _LimbDarkenedMap(
    CustomBase, starry.maps.LimbDarkenedBase, starry.maps.MapBase
):
    __doc__ = starry.maps.LimbDarkenedBase.__doc__


class _ReflectedLightMap(
    CustomBase, starry.maps.ReflectedBase, starry.maps.MapBase
):
    __doc__ = starry.maps.ReflectedBase.__doc__


class _RadialVelocityMap(CustomBase, starry.maps.RVBase, starry.maps.MapBase):
    __doc__ = starry.maps.RVBase.__doc__


starry._Map = _Map
starry._LimbDarkenedMap = _LimbDarkenedMap
starry._ReflectedLightMap = _ReflectedLightMap
starry._RadialVelocityMap = _RadialVelocityMap
autoclass_content = "both"


# -- Project information -----------------------------------------------------

project = "starry"
copyright = "2019, Rodrigo Luger"
author = "Rodrigo Luger"

# The full version, including alpha/beta/rc tags
version = starry.__version__
release = starry.__version__

# Get current git branch
branch = os.getenv("GHBRANCH", "master")

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

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]
html_js_files = ["js/version.js", "js/fix_class_names.js"]
html_css_files = ["css/hide_input.css"]

# -- Extension settings ------------------------------------------------------


html_js_files += [DEFAULT_EMBED_REQUIREJS_URL]

# Add a heading to notebooks (TODO: switch to `master`)
nbsphinx_prolog = """
{%s set docname = env.doc2path(env.docname, base=None) %s}
.. note:: This tutorial was generated from a Jupyter notebook that can be
          downloaded `here <https://github.com/rodluger/starry/blob/%s/{{ docname }}>`_.
""" % (
    "%",
    "%",
    branch,
)
nbsphinx_prompt_width = 0
nbsphinx_timeout = 600
napoleon_use_ivar = True
todo_include_todos = True
autosummary_generate = True
autodoc_docstring_signature = True
