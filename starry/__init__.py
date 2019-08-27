# -*- coding: utf-8 -*-
__version__ = "1.0.0.dev4"

# Was `starry` imported from setup.py?
try:
    __STARRY_SETUP__
except NameError:
    __STARRY_SETUP__ = False

# Is this a docs run?
try:
    __STARRY_DOCS__
except NameError:
    __STARRY_DOCS__ = False


# Import all modules
if not __STARRY_SETUP__:

    # Import the main interface
    from .maps import Map

    # Force double precision
    import theano.tensor as tt

    tt.config.floatX = "float64"
