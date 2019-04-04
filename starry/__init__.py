# -*- coding: utf-8 -*-
__version__ = '1.0.0.dev0'

# Was `starry` imported from setup.py?
try:
    __STARRY_SETUP__
except NameError:
    __STARRY_SETUP__ = False

# Define the modules and their bitsums
modules = dict(
    _STARRY_DEFAULT_YLM_DOUBLE_=1,
    _STARRY_DEFAULT_LD_DOUBLE_=2,
    _STARRY_SPECTRAL_YLM_DOUBLE_=4,
    _STARRY_SPECTRAL_LD_DOUBLE_=8,
    _STARRY_TEMPORAL_YLM_DOUBLE_=16,
    _STARRY_DEFAULT_REFLECTED_DOUBLE_=32,
    _STARRY_SPECTRAL_REFLECTED_DOUBLE_=64,
    _STARRY_TEMPORAL_REFLECTED_DOUBLE_=128,
    _STARRY_DEFAULT_YLM_MULTI_=256,
    _STARRY_DEFAULT_LD_MULTI_=512,
    _STARRY_SPECTRAL_YLM_MULTI_=1024,
    _STARRY_SPECTRAL_LD_MULTI_=2048,
    _STARRY_TEMPORAL_YLM_MULTI_=4096,
    _STARRY_DEFAULT_REFLECTED_MULTI_=8192,
    _STARRY_SPECTRAL_REFLECTED_MULTI_=16384,
    _STARRY_TEMPORAL_REFLECTED_MULTI_=32768,
    _STARRY_EXTENSIONS_=65536
)

# Import all modules
if not __STARRY_SETUP__:
    from . import kepler
    from .extensions import *
    from . import ops
    from .map import Map