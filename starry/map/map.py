# -*- coding: utf-8 -*-
from .pybase import PythonMapBase
from .. import modules


__all__ = ["Map"]


def module_not_found(name):
    """Throw an error if a module is not found."""
    bit = modules[name]
    raise ModuleNotFoundError("Requested module not found. " + 
        "Please re-compile `starry` with bit %d enabled." % bit)


def Map(ydeg, udeg=0, **kwargs):
    """
    Figures out which `Map` class the user wants and instantiates it.

    """
    # Figure out the correct base class
    multi = kwargs.pop('multi', False)
    reflected = kwargs.pop('reflected', False)
    nw = kwargs.pop('nw', None)
    nt = kwargs.pop('nt', None)
    if (ydeg == 0) and (udeg > 0):
        limbdarkened = True
        kwargs["udeg"] = udeg
        if nt is not None:
            raise NotImplementedError("Pure limb darkening is not implemented " + 
                                      "for temporal maps.")
    else:
        limbdarkened = False
        kwargs["ydeg"] = ydeg
        kwargs["udeg"] = udeg
    if (nw is None):
        if (nt is None):
            if (not reflected):
                if (not multi):
                    if (limbdarkened):
                        try:
                            from .._starry_default_limbdarkened_double import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_DEFAULT_LIMBDARKENED_DOUBLE_")
                    else:
                        try:
                            from .._starry_default_double import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_DEFAULT_DOUBLE_")
                else:
                    if (limbdarkened):
                        try:
                            from .._starry_default_limbdarkened_multi import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_DEFAULT_LIMBDARKENED_MULTI_")
                    else:
                        try:
                            from .._starry_default_multi import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_DEFAULT_MULTI_")
            else:
                if (not multi):
                    try:
                        from .._starry_default_reflected_double import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_DEFAULT_REFLECTED_DOUBLE_")
                else:
                    try:
                        from .._starry_default_reflected_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_DEFAULT_REFLECTED_MULTI_")
        else:
            kwargs['nterms'] = nt
            if (not reflected):
                if (not multi):
                    try:
                        from .._starry_temporal_double import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_TEMPORAL_DOUBLE_")
                else:
                    try:
                        from .._starry_temporal_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_TEMPORAL_MULTI_")
            else:
                if (not multi):
                    try:
                        from .._starry_temporal_reflected_double import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_TEMPORAL_REFLECTED_DOUBLE_")
                else:
                    try:
                        from .._starry_temporal_reflected_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_TEMPORAL_REFLECTED_MULTI_")
    else:
        if (nt is None):
            kwargs['nterms'] = nw
            if (not reflected):
                if (not multi):
                    if (limbdarkened):
                        try:
                            from .._starry_spectral_limbdarkened_double import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_SPECTRAL_LIMBDARKENED_DOUBLE_")
                    else:
                        try:
                            from .._starry_spectral_double import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_SPECTRAL_DOUBLE_")
                else:
                    if (limbdarkened):
                        try:
                            from .._starry_spectral_limbdarkened_multi import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_SPECTRAL_LIMBDARKENED_MULTI_")
                    else:
                        try:
                            from .._starry_spectral_multi import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_SPECTRAL_MULTI_")
            else:
                if (not multi):
                    try:
                        from .._starry_spectral_reflected_double import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_SPECTRAL_REFLECTED_DOUBLE_")
                else:
                    try:
                        from .._starry_spectral_reflected_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_SPECTRAL_REFLECTED_MULTI_")
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
            self._limbdarkened = limbdarkened
            super(Map, self).__init__(*init_args, **init_kwargs)

    # Hack this function's docstring
    __doc__ = Map.__doc__

    # Return an instance
    return Map(**kwargs)