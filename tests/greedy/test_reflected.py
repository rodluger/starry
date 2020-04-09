# -*- coding: utf-8 -*-
"""
Test reflected light calculations.

"""
import starry
import numpy as np
import matplotlib.pyplot as plt


def test_one_over_r_squared(n_tests=10, plot=False):
    map = starry.Map(2, reflected=True)
    flux0 = map.flux()
    zs = np.linspace(1, 10, 100)
    flux = map.flux(xs=0, ys=0, zs=zs)

    if plot:
        plt.plot(zs, flux)
        plt.plot(zs, flux0 / zs ** 2)
        plt.show()

    assert np.allclose(flux, flux0 / zs ** 2)
