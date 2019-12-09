# -*- coding: utf-8 -*-
from .starry_version import __version__


# Store the package directory
import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))


# Force double precision
import theano.tensor as tt

tt.config.floatX = "float64"


# Set up the default config
from .configdefaults import Config

config = Config()


# Import the main interface
from . import constants, core, indices, kepler, maps, sht, plotting
from .maps import Map
from .kepler import Primary, Secondary, System


# Clean up the namespace
del tt
del Config


__all__ = [
    "__version__",
    "constants",
    "core",
    "indices",
    "kepler",
    "maps",
    "sht",
    "plotting",
    "Map",
    "Primary",
    "Secondary",
    "System",
]
