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
            raise NotImplementedError("Limb darkening is not implemented " + 
                                      "for temporal maps.")
        # TODO: Implement spectral limb darkening!
        if nw is not None:
            raise NotImplementedError("Limb darkening has not yet been " + 
                                      "implemented for spectral maps.")
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
                            from .._starry_limbdarkened_double import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_LIMBDARKENED_DOUBLE_")
                    else:
                        try:
                            from .._starry_default_double import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_DEFAULT_DOUBLE_")
                else:
                    if (limbdarkened):
                        try:
                            from .._starry_limbdarkened_multi import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_LIMBDARKENED_MULTI_")
                    else:
                        try:
                            from .._starry_default_multi import Map as CMapBase
                        except ModuleNotFoundError:
                            module_not_found("_STARRY_DEFAULT_MULTI_")
            else:
                if (not multi):
                    try:
                        from .._starry_default_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_DEFAULT_REFL_DOUBLE_")
                else:
                    try:
                        from .._starry_default_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_DEFAULT_REFL_MULTI_")
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
                        from .._starry_temporal_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_TEMPORAL_REFL_DOUBLE_")
                else:
                    try:
                        from .._starry_temporal_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_TEMPORAL_REFL_MULTI_")
    else:
        if (nt is None):
            kwargs['nterms'] = nw
            if (not reflected):
                if (not multi):
                    try:
                        from .._starry_spectral_double import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_SPECTRAL_DOUBLE_")
                else:
                    try:
                        from .._starry_spectral_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_SPECTRAL_MULTI_")
            else:
                if (not multi):
                    try:
                        from .._starry_spectral_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_SPECTRAL_REFL_DOUBLE_")
                else:
                    try:
                        from .._starry_spectral_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        module_not_found("_STARRY_SPECTRAL_REFL_MULTI_")
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