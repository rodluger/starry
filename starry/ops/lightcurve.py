# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import theano.tensor as tt
from .theano_op import DopplerMapOp


# TODO TODO TODO TODO This routine needs a lot of work


__all__ = ["LightCurve"]


class LightCurve(object):

    """A light curve computed using starry"""

    __citations__ = ("starry", )

    def __init__(self, ydeg=0, udeg=0):
        self.starry_op = DopplerMapOp(ydeg=ydeg, udeg=udeg)

    def get_light_curve(self, y=None, orbit=None, r=None, t=None,
                        theta0=0.0, t0=0.0, per=None):
        """Get the light curve for an orbit at a set of times"""
        if y is None:
            raise ValueError("missing required argument 'y'")
        if orbit is None:
            raise ValueError("missing required argument 'orbit'")
        if r is None:
            raise ValueError("missing required argument 'r'")
        if t is None:
            raise ValueError("missing required argument 't'")

        y = tt.as_tensor_variable(y)
        r = tt.as_tensor_variable(r)
        t = tt.as_tensor_variable(t)

        # TODO: Why is this necessary?
        r = tt.reshape(r, (r.size,))

        # TODO: Add exposure time integration
        tgrid = t
        rgrid = r
        
        # Get coords on plane of sky
        # TODO: Ensure our conventions agree
        coords = orbit.get_relative_position(tgrid)

        # TODO: Is this reshape necessary? (For > 1 planets I think)
        #xo = tt.reshape(coords[0], rgrid.shape)
        #yo = tt.reshape(coords[1], rgrid.shape)
        #zo = tt.reshape(-coords[2], rgrid.shape)

        xo = coords[0]
        yo = coords[1]
        zo = -coords[2]

        # Figure out rotational state
        # TODO: Add axis arg
        theta = tt.ones_like(xo) * theta0
        if per is not None:
            theta += (2 * np.pi / per * (t - t0)) % (2 * np.pi)
        theta = tt.as_tensor_variable(theta)

        lc = self._compute_light_curve(
            y,
            theta, 
            xo/orbit.r_star, 
            yo/orbit.r_star, 
            zo/orbit.r_star,
            rgrid/orbit.r_star
        )
        return lc

    def _compute_light_curve(self, y, theta, xo, yo, zo, ro):
        """Compute the light curve"""
        return self.starry_op(y, theta, xo, yo, zo, ro)