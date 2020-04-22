# -*- coding: utf-8 -*-
from astropy import constants, units
import numpy as np

# Miscelaneous status codes
STARRY_ORTHOGRAPHIC_PROJECTION = 0
STARRY_RECTANGULAR_PROJECTION = 1
STARRY_MOLLWEIDE_PROJECTION = 2
STARRY_COVARIANCE_SCALAR = 0
STARRY_COVARIANCE_VECTOR = 1
STARRY_COVARIANCE_MATRIX = 2
STARRY_COVARIANCE_CHOLESKY = 3

# Gravitational constant in internal units
G_grav = constants.G.to(units.R_sun ** 3 / units.M_sun / units.day ** 2).value

# The normalization of maps in reflected light is
# a little different than those in emitted light.
# This factor ensures the normalization of the flux
# is correct for a Lambertian sphere.
STARRY_REFLECTANCE_FLUX_NORMALIZATION = 1.0 / np.pi
