# -*- coding: utf-8 -*-
"""
Test reflected light calculations.

"""
import starry
import numpy as np


def test_one_over_r_squared(n_tests=10):
    map = starry.Map(2, reflected=True)
    flux0 = map.flux()
    zo = np.linspace(1, 10, 100)
    flux = map.flux(xo=0, yo=0, zo=zo)
    assert np.allclose(flux, flux0 / zo ** 2)
