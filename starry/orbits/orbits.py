# -*- coding: utf-8 -*-
import exoplanet
import theano.tensor as tt
import numpy as np
from ..ops import autocompile


class KeplerianOrbit(exoplanet.orbits.KeplerianOrbit):
    """
    A wrapper around `exoplanet.orbits.KeplerianOrbit` that
    plays nice with `starry`. Refer to the docs of that class
    for all accepted keywords. In addition to those, this class
    accepts the following keyword arguments:

    Args:
        r_planet: The radius of the planet in ``R_sun``. Default is 
            the radius of the Earth.
        rot_period: The period of rotation of the planet in days.
            Default ``1.0``. Set to ``None`` to disable rotation.
        theta0: The rotational phase in degrees at ``t=t0``.
            Default ``0.0``
        lazy: 

    """

    def __init__(self, r_planet=0.0091577, rot_period=1.0, theta0=0.0, **kwargs):
        super(KeplerianOrbit, self).__init__(**kwargs)
        self.lazy = True
        self.r_planet = tt.as_tensor_variable(r_planet).astype(tt.config.floatX)
        if rot_period == 0.0 or rot_period is None:
            self._is_rotating = False
            rot_period = 0.0
        else:
            self._is_rotating = True
        self.rot_period = tt.as_tensor_variable(rot_period).astype(tt.config.floatX)
        self.theta0 = tt.as_tensor_variable(theta0).astype(tt.config.floatX)
        assert (self.period.ndim == 0), \
            "Only single-body systems are currently supported."
    
    @autocompile("_get_occultation_coords", tt.dvector())
    def _get_occultation_coords(self, t):
        """
        TODO: Add exposure time integration.

        """
        # Get the relative position of the central body (star) and the
        # orbiting body (planet) in units of R_sun.
        coords = self.get_relative_position(t)

        # Convert to units of the planet radius
        xo = -coords[0] / self.r_planet
        yo = -coords[1] / self.r_planet
        # TODO: This convention may change in the next `exoplanet` release
        zo = coords[2] / self.r_planet

        # Rotational phase
        theta = self.theta0
        if self._is_rotating:
            theta += 360.0 / self.rot_period * (t - self.t0)

        # Star radius in units of planet radius
        ro = self.r_star / self.r_planet

        return xo, yo, zo, ro, theta