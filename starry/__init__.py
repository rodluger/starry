# -*- coding: utf-8 -*-
from .starry_version import __version__


# Store the package directory
import os

_PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))


# Force double precision
import theano.tensor as tt

tt.config.floatX = "float64"


# Import the main interface
from ._config import config
from . import kepler, linalg, maps
from .maps import Map
from .kepler import Primary, Secondary, System


# Clean up the namespace
del tt
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
