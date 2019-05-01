# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["FilterBase"]


class Filter(object):
    def __init__(self, parent):
        self.parent = parent
    def __getitem__(self, inds):
        return self.parent._get_filter(inds)
    def __setitem__(self, inds, vals):
        return self.parent._set_filter(inds, vals)
    def __repr__(self):
        return "<starry.Map.filter>"


class FilterBase(object):
    """
    .. autoattribute:: filter
    """

    @property
    def filter(self):
        """
        The filter coefficients. 
        
        These are a vector of spherical harmonic coefficients that are
        applied as a static multiplicative filter to the map whenever
        intensities or fluxes or computed. This filter is the generalization
        of limb darkening to non-radially symmetric fields. It can therefore
        be used to model processes such as non-radial limb darkening, 
        gravity darkening, clouds, etc.

        The filter coefficients can be accessed and set via the item
        getter/setter operator :py:obj:`[]`. For instance, the following
        line

        .. code-block:: python

            map.filter[1, 0] = 1.0

        sets the :math:`Y_{1, 0}` filter coefficient. Regardless of the orientation
        of the map, it will always be weighted by this spherical harmonic
        when computing intensities or fluxes. In this example, the effect is
        the same as that of linear limb darkening.

        See :py:meth:`__setitem__` and :py:meth:`__getitem__` for details. 
        """
        return Filter(self)