# -*- coding: utf-8 -*-
"""
Functions to test limb darkened occultations.

"""
import starry
import numpy as np


def test_quadratic():
    """Test quadratic limb darkening."""
    # Occultation params
    xo = np.linspace(-1.5, 1.5, 1000)
    yo = 0.3
    ro = 0.1

    # LD params
    u1 = 0.5
    u2 = 0.25

    # Limb-darkened map
    map1 = starry.Map(udeg=2)
    map1[1] = u1
    map1[2] = u2
    flux1 = map1.flux(xo=xo, yo=yo, ro=ro)

    # Emulate it with a Ylm map (Equation 38 in Luger et al. 2019)
    map2 = starry.Map(ydeg=2)
    y00 = 2.0 * np.sqrt(np.pi) / 3.0 * (3.0 - 3.0 * u1 - 4.0 * u2)
    y10 = 2.0 * np.sqrt(np.pi / 3.0) * (u1 + 2.0 * u2)
    y20 = -4.0 / 3.0 * np.sqrt(np.pi / 5.0) * u2
    y = np.array([y00, 0, y10, 0, 0, 0, y20, 0, 0]) / (
        np.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    )
    flux2 = map2.design_matrix(xo=xo, yo=yo, ro=ro).dot(y)
    flux2 *= np.sqrt(np.pi) / 2  # add in the starry normalization

    assert np.allclose(flux1, flux2)
