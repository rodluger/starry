# -*- coding: utf-8 -*-
from .pybase import PythonMapBase
from .. import modules


__all__ = ["Map"]


def Map(*args, **kwargs):
    """
    Figures out which `Map` class the user wants and instantiates it.

    """
    # Figure out the correct base class
    multi = kwargs.pop('multi', False)
    reflected = kwargs.pop('reflected', False)
    nw = kwargs.pop('nw', None)
    nt = kwargs.pop('nt', None)
    if (nw is None):
        if (nt is None):
            if (not reflected):
                if (not multi):
                    try:
                        from .._starry_default_double import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_DEFAULT_DOUBLE_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
                else:
                    try:
                        from .._starry_default_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_DEFAULT_MULTI_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
            else:
                if (not multi):
                    try:
                        from .._starry_default_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_DEFAULT_REFL_DOUBLE_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
                else:
                    try:
                        from .._starry_default_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_DEFAULT_REFL_MULTI_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
        else:
            kwargs['nterms'] = nt
            if (not reflected):
                if (not multi):
                    try:
                        from .._starry_temporal_double import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_TEMPORAL_DOUBLE_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
                else:
                    try:
                        from .._starry_temporal_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_TEMPORAL_MULTI_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
            else:
                if (not multi):
                    try:
                        from .._starry_temporal_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_TEMPORAL_REFL_DOUBLE_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
                else:
                    try:
                        from .._starry_temporal_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_TEMPORAL_REFL_MULTI_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
    else:
        if (nt is None):
            kwargs['nterms'] = nw
            if (not reflected):
                if (not multi):
                    try:
                        from .._starry_spectral_double import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_SPECTRAL_DOUBLE_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
                else:
                    try:
                        from .._starry_spectral_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_SPECTRAL_MULTI_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
            else:
                if (not multi):
                    try:
                        from .._starry_spectral_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_SPECTRAL_REFL_DOUBLE_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
                else:
                    try:
                        from .._starry_spectral_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        bit = modules["_STARRY_SPECTRAL_REFL_MULTI_"]
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit %d enabled." % bit)
        else:
            raise ValueError("Spectral maps cannot have temporal variability.")


    # Subclass it
    class Map(CMapBase, PythonMapBase):
        __doc__ = CMapBase.__doc__
        def __init__(self, *init_args, **init_kwargs):
            self._multi = multi
            self._reflected = reflected
            self._temporal = (nt is not None)
            self._spectral = (nw is not None)
            super(Map, self).__init__(*init_args, **init_kwargs)

    # Hack this function's docstring
    __doc__ = Map.__doc__

    # Return an instance
    return Map(*args, **kwargs)