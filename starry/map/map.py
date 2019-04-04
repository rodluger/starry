# -*- coding: utf-8 -*-
from .pybase import PythonMapBase
from .. import modules


__all__ = ["Map"]


def import_by_name(name):
    """Import a module by name."""
    name = "_starry_" + name
    try:
        exec("from ..%s import Map as CMapBase" % name, globals())
    except ModuleNotFoundError:
        bit = modules[(name + "_").upper()]
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
    spectral = (nw is not None)
    nt = kwargs.pop('nt', None)
    temporal = (nt is not None)
    if (ydeg == 0) and (udeg > 0):
        limbdarkened = True
        kwargs["udeg"] = udeg
    else:
        limbdarkened = False
        kwargs["ydeg"] = ydeg
        kwargs["udeg"] = udeg

    # Disallowed combinations
    if limbdarkened and temporal:
        raise NotImplementedError("Pure limb darkening is not implemented " + 
                                  "for temporal maps.")
    elif limbdarkened and reflected:
        raise NotImplementedError("Pure limb darkening is not implemented " + 
                                  "in reflected light.") 
    elif spectral and temporal:
        raise NotImplementedError("Spectral maps cannot have time dependence.")

    # Figure out the module flags
    if spectral:
        kind = "spectral"
    elif temporal:
        kind = "temporal"
    else:
        kind = "default"
    if limbdarkened:
        flag = "ld"
    elif reflected:
        flag = "reflected"
    else:
        flag = "ylm"
    if multi:
        dtype = "multi"
    else:
        dtype = "double"

    # Import it
    import_by_name('%s_%s_%s' % (kind, flag, dtype))

    # Subclass it
    class Map(CMapBase, PythonMapBase):
        __doc__ = CMapBase.__doc__
        def __init__(self, *init_args, **init_kwargs):
            self._multi = multi
            self._reflected = reflected
            self._temporal = temporal
            self._spectral = spectral
            self._limbdarkened = limbdarkened
            super(Map, self).__init__(*init_args, **init_kwargs)

    # Hack this function's docstring
    __doc__ = Map.__doc__

    # Return an instance
    return Map(**kwargs)