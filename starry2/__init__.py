# -*- coding: utf-8 -*-
__version__ = None
__mapdoc__ = None

# Import Python routines
from . import _plotting
from . import kepler
from .extensions import *

# Theano
try:
    from .theano_op import starry_op
except ImportError:
    starry_op = None

# Import all available C++ modules
try:
    from . import _starry_default_double
    if __version__ is None:
        from ._starry_default_double import __version__
        __mapdoc__ = _starry_default_double.Map.__doc__
except ImportError:
    _starry_default_double = None

try:
    from . import _starry_default_multi
    if __version__ is None:
        from ._starry_default_multi import __version__
        __mapdoc__ = _starry_default_multi.Map.__doc__
except ImportError:
    _starry_default_multi = None

try:
    from . import _starry_spectral_double
    if __version__ is None:
        from ._starry_spectral_double import __version__
        __mapdoc__ = _starry_spectral_double.Map.__doc__
except ImportError:
    _starry_spectral_double = None

try:
    from . import _starry_spectral_multi
    if __version__ is None:
        from ._starry_spectral_multi import __version__
        __mapdoc__ = _starry_spectral_multi.Map.__doc__
except ImportError:
    _starry_spectral_multi = None

try:
    from . import _starry_temporal_double
    if __version__ is None:
        from ._starry_temporal_double import __version__
        __mapdoc__ = _starry_temporal_double.Map.__doc__
except ImportError:
    _starry_temporal_double = None

try:
    from . import _starry_temporal_multi
    if __version__ is None:
        from ._starry_temporal_multi import __version__
        __mapdoc__ = _starry_temporal_multi.Map.__doc__
except ImportError:
    _starry_temporal_multi = None


# Class factory
def Map(lmax=2, nw=None, nt=None, multi=False):
    if (nw is None) and (nt is None) and (not multi):
        if _starry_default_double is not None:
            return _starry_default_double.Map(lmax)
        else:
            raise ImportError("Module not available: _STARRY_DEFAULT_DOUBLE_.")
    elif (nw is None) and (nt is None) and (multi):
        if _starry_default_multi is not None:
            return _starry_default_multi.Map(lmax)
        else:
            raise ImportError("Module not available: _STARRY_DEFAULT_MULTI_.")
    elif (nw is not None) and (nw > 0) and (nt is None) and (not multi):
        if _starry_spectral_double is not None:
            return _starry_spectral_double.Map(lmax, nw)
        else:
            raise ImportError("Module not available: _STARRY_SPECTRAL_DOUBLE_.")
    elif (nw is not None) and (nw > 0) and (nt is None) and (multi):
        if _starry_spectral_multi is not None:
            return _starry_spectral_multi.Map(lmax, nw)
        else:
            raise ImportError("Module not available: _STARRY_SPECTRAL_MULTI_.")
    elif (nt is not None) and (nt > 0) and (nw is None) and (not multi):
        if _starry_temporal_double is not None:
            return _starry_temporal_double.Map(lmax, nt)
        else:
            raise ImportError("Module not available: _STARRY_TEMPORAL_DOUBLE_.")
    elif (nt is not None) and (nt > 0) and (nw is None) and (multi):
        if _starry_temporal_multi is not None:
            return _starry_temporal_multi.Map(lmax, nt)
        else:
            raise ImportError("Module not available: _STARRY_TEMPORAL_MULTI_.")
    else:
        raise ValueError("Invalid argument(s) to `Map`.")


# Hack the docstring
Map.__doc__ = __mapdoc__