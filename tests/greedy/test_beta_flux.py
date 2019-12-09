# -*- coding: utf-8 -*-
"""
Functions to compare stuff to the beta version.

"""
import starry
import starry_beta
import numpy as np
import pytest
import itertools


# Setup
map = starry.Map(ydeg=1, udeg=2)
map_beta = starry_beta.Map(3)
y = [[0, 0, 0], [1, 1, 1]]
u = [[0, 0], [0.5, 0.25]]
theta = [0, np.linspace(-180, 180, 30)]
xo = [0, np.linspace(-1.5, 1.5, 30)]
yo = [0.1]
ro = [0, 0.1, 2.0]
axis = [[0, 1, 0]]  # TODO: Test different axes; need to align the maps!
params = itertools.product(y, u, theta, xo, yo, ro, axis)


@pytest.mark.parametrize("y,u,theta,xo,yo,ro,axis", params)
def test_flux(y, u, theta, xo, yo, ro, axis):
    # NOTE: Limb-darkened Ylm maps are bugged in the beta version
    if np.any(y) and np.any(u):
        return
    map[1, :] = y
    map[1], map[2] = u
    map.axis = axis
    map_beta[1, :] = y
    map_beta[1], map_beta[2] = u
    map_beta.axis = axis
    assert np.allclose(
        map.flux(theta=theta, xo=xo, yo=yo, ro=ro),
        map_beta.flux(theta=theta, xo=xo, yo=yo, ro=ro),
    )
