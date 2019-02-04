# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import theano.tensor as tt
from .theano_op import TheanoOp


__all__ = ["LightCurve"]


class LightCurve(object):

    """A light curve computed using starry"""

    __citations__ = ("starry", )

    def __init__(self, lmax, model=None):
        self.starry_op = TheanoOp(lmax)

    def get_light_curve(self, y=None, u=None, orbit=None, r=None, t=None,
        theta0=0.0, t0=0.0, per=None):
        """Get the light curve for an orbit at a set of times"""
        if y is None:
            raise ValueError("missing required argument 'y'")
        if u is None:
            raise ValueError("missing required argument 'u'")
        if orbit is None:
            raise ValueError("missing required argument 'orbit'")
        if r is None:
            raise ValueError("missing required argument 'r'")
        if t is None:
            raise ValueError("missing required argument 't'")

        y = tt.as_tensor_variable(y)
        u = tt.as_tensor_variable(u)
        r = tt.as_tensor_variable(r)
        r = tt.reshape(r, (r.size,))
        t = tt.as_tensor_variable(t)

        # TODO: Add exposure time integration
        tgrid = t
        rgrid = r
        
        # Get coords on plane of sky
        coords = orbit.get_relative_position(tgrid)
        xo = tt.reshape(coords[0], rgrid.shape)
        yo = tt.reshape(coords[1], rgrid.shape)
        zo = tt.reshape(coords[2], rgrid.shape)

        # Figure out rotational state
        # TODO: Add axis arg
        theta = tt.ones_like(xo) * theta0
        if per is not None:
            theta += (2 * np.pi / per * (t - t0)) % (2 * np.pi)

        lc = self._compute_light_curve(
            y,
            u,
            theta, 
            xo/orbit.r_star, 
            yo/orbit.r_star, 
            rgrid/orbit.r_star, 
            zo/orbit.r_star
        )
        return lc

    def _compute_light_curve(self, y, u, theta, xo, yo, ro, zo=None):
        """Compute the light curve"""
        if zo is None:
            zo = -tt.ones_like(xo)
        return self.starry_op(y, u, theta, xo, yo, ro, zo)