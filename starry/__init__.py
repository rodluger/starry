# -*- coding: utf-8 -*-
from .starry_version import __version__


# Store the package directory
import os

_PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))


# Theano compatibility hacks
from . import compat


# Import the main interface
from ._config import config
from . import kepler, linalg, maps
from .maps import Map
from .kepler import Primary, Secondary, System


# Clean up the namespace
del os


__all__ = [
    "__version__",
    "kepler",
    "linalg",
    "maps",
    "Map",
    "Primary",
    "Secondary",
    "System",
]
