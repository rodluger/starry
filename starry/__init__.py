# -*- coding: utf-8 -*-
from ._starry_mono_64 import __version__
from . import _starry_mono_64, _starry_mono_128, \
              _starry_spectral_64, _starry_spectral_128
from . import kepler


# Class factory
def Map(lmax=2, nwav=1, multi=False):
    if (nwav == 1) and (not multi):
        return _starry_mono_64.Map(lmax, nwav)
    elif (nwav == 1) and (multi):
        return _starry_mono_128.Map(lmax, nwav)
    elif (nwav > 1) and (not multi):
        return _starry_spectral_64.Map(lmax, nwav)
    elif (nwav > 1) and (multi):
        return _starry_spectral_128.Map(lmax, nwav)
    else:
        raise ValueError("Invalid argument(s) to `Map`.")


# Hack the docstring
Map.__doc__ = _starry_mono_64.Map.__doc__
