# -*- coding: utf-8 -*-
__version__ = "1.0.0.dev6"

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

    # Force double precision
    import theano.tensor as tt

    tt.config.floatX = "float64"

    # Set up the default config
    from .configdefaults import Config

    config = Config()

    # Import the main interface
    from . import indices, kepler, maps, sht, utils
    from . import extensions
    from .maps import Map
    from .kepler import Primary, Secondary, System

    # Clean up the namespace
    del tt
    del Config
