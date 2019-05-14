# -*- coding: utf-8 -*-
__version__ = '1.0.0.dev2'

# Was `starry` imported from setup.py?
try:
    __STARRY_SETUP__
except NameError:
    __STARRY_SETUP__ = False

# Import all modules
if not __STARRY_SETUP__:
    from .Map import Map