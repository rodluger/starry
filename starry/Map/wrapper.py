# -*- coding: utf-8 -*-
from .bases import *
from .. import modules


__all__ = ["Map"]


def import_by_name(name):
    """Import a module by name."""
    name = "_starry_" + name
    try:
        exec("from ..%s import Map as CBase" % name, globals())
    except ModuleNotFoundError:
        bit = modules[(name + "_").upper()]
        raise ModuleNotFoundError("Requested module not found. " + 
            "Please re-compile `starry` with bit %d enabled." % bit)


def Map(ydeg=0, udeg=0, fdeg=0, nt=None, nw=None, 
        reflected=False, doppler=False, multi=False):
    """
    The main interface to ``starry``, representing a generalized 
    celestial body surface map.
    
    Keyword Arguments:
        ydeg (int): The highest spherical harmonic degree of the map.
        udeg (int): The highest limb darkening degree of the map.
        fdeg (int): The highest degree of the custom multiplicative 
            filter applied to the map.
        nt (int): The number of temporal components in the map. 
            Cannot be set simultaneously with ``nw``.
        nw (int): The number of spectral components in the map. 
            Cannot be set simultaneously with ``nt``.
        reflected (bool): If ``True``, performs all calculations in 
            reflected light. The spherical harmonic expansion now 
            corresponds to the *albedo* of the surface
        doppler (bool): If ``True``, enables Doppler mode. 
            See :py:class:`DopplerMap` for details.
        multi (bool): If ``True``, performs all calculations using 
            multi-precision floating point arithmetic. The number of 
            digits of the multi-precision type is controlled by the 
            ``STARRY_NMULTI`` compile-time constant.

    Instances returned by ``Map`` are spherical bodies whose surfaces are 
    described by spherical harmonics. In the default case, a vector
    of spherical harmonic coefficients describes the specific intensity
    everywhere on the surface, although it may also describe the
    albedo (for maps in reflected light) or the brightness-weighted
    radial velocity (for Doppler maps). ``Map`` allows
    users to easily and efficiently manipulate the surface representation
    and compute intensities and fluxes (light curves) as the object
    rotates and becomes occulted by other spherical bodies.
    """
    # Figure out the correct base class
    spectral = (nw is not None)
    temporal = (nt is not None)
    map_kwargs = {}

    if doppler:
        if reflected:
            raise NotImplementedError("Doppler maps are not implemented in reflected light.")
        limbdarkened = False
        map_kwargs["ydeg"] = ydeg
        map_kwargs["udeg"] = udeg
        map_kwargs["fdeg"] = 3
    elif (ydeg == 0) and (fdeg == 0) and (udeg > 0):
        limbdarkened = True
        map_kwargs["udeg"] = udeg
    else:
        limbdarkened = False
        map_kwargs["ydeg"] = ydeg
        map_kwargs["udeg"] = udeg
        map_kwargs["fdeg"] = fdeg

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
        map_kwargs["nterms"] = nw
    elif temporal:
        kind = "temporal"
        map_kwargs["nterms"] = nt
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

    # Figure out the base classes
    bases = (CBase,)
    if doppler:
        bases = (DopplerBase, YlmBase,) + bases
    elif limbdarkened:
        bases = (LimbDarkenedBase,) + bases
    else:
        bases = (YlmBase,) + bases
        if (fdeg > 0):
            bases = (FilterBase,) + bases

    # Subclass it
    class Map(*bases):
        __doc__ = "".join([base.__doc__ if base.__doc__ is not None else "" 
                           for base in bases])
        def __init__(self, **kwargs):
            self._multi = multi
            self._reflected = reflected
            self._temporal = temporal
            self._spectral = spectral
            self._scalar = not (self._temporal or self._spectral)
            self._limbdarkened = limbdarkened
            super(Map, self).__init__(**kwargs)

    # Return an instance
    return Map(**map_kwargs)