# -*- coding: utf-8 -*-
__version__ = '1.0.0.dev0'

# Was `starry` imported from setup.py?
try:
    __STARRY_SETUP__
except NameError:
    __STARRY_SETUP__ = False

# Define the modules and their bitsums
modules = dict(
    _STARRY_DEFAULT_DOUBLE_=1,
    _STARRY_DEFAULT_MULTI_=2,
    _STARRY_SPECTRAL_DOUBLE_=4,
    _STARRY_SPECTRAL_MULTI_=8,
    _STARRY_TEMPORAL_DOUBLE_=16,
    _STARRY_TEMPORAL_MULTI_=32,
    _STARRY_DEFAULT_REFL_DOUBLE_=64,
    _STARRY_DEFAULT_REFL_MULTI_=128,
    _STARRY_SPECTRAL_REFL_DOUBLE_=256,
    _STARRY_SPECTRAL_REFL_MULTI_=512,
    _STARRY_TEMPORAL_REFL_DOUBLE_=1024,
    _STARRY_TEMPORAL_REFL_MULTI_=2048,
    _STARRY_LIMBDARKENED_DOUBLE_=4096,
    _STARRY_LIMBDARKENED_MULTI_=8192,
    _STARRY_EXTENSIONS_=16384
)

# Import all modules
if not __STARRY_SETUP__:
    from . import kepler
    from .extensions import *
    from . import ops
    from .map import Map