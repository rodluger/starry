# -*- coding: utf-8 -*-
"""
Linear solve / likelihood tests.

"""
import starry
import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import multivariate_normal
import pytest
import itertools


vals = ["scalar", "vector", "matrix", "cholesky"]
inputs = itertools.product(vals, vals)


@pytest.mark.parametrize("L,C", inputs)
def test_lnlike(L, C):
    """Test the log marginal likelihood method."""
    # Instantiate a dipole map in reflected light
    map = starry.Map(ydeg=1, reflected=True)
    map.inc = 60
    y_true = [0.1, 0.2, 0.3]
    map[1, :] = y_true

    # Generate a synthetic light curve with just a little noise
    theta = np.linspace(0, 360, 100)
    phi = 3.5 * theta
    xo = np.cos(phi * np.pi / 180)
    yo = 0.1 * np.cos(phi * np.pi / 180)
    zo = np.sin(phi * np.pi / 180)
    kwargs = dict(theta=theta, xo=xo, yo=yo, zo=zo)
    flux = map.flux(**kwargs)
    sigma = 1e-5
    np.random.seed(0)
    flux += np.random.randn(len(theta)) * sigma

    # Place a generous prior on the map coefficients
    if L == "scalar":
        map.set_prior(L=1)
    elif L == "vector":
        map.set_prior(L=np.ones(map.Ny - 1))
    elif L == "matrix":
        map.set_prior(L=np.eye(map.Ny - 1))
    elif L == "cholesky":
        map.set_prior(cho_L=np.eye(map.Ny - 1))

    # Provide the dataset
    if C == "scalar":
        map.set_data(flux, C=sigma ** 2)
    elif C == "vector":
        map.set_data(flux, C=np.ones(len(flux)) * sigma ** 2)
    elif C == "matrix":
        map.set_data(flux, C=np.eye(len(flux)) * sigma ** 2)
    elif C == "cholesky":
        map.set_data(flux, cho_C=np.eye(len(flux)) * sigma)

    # Compute the marginal log likelihood for different inclinations
    incs = [15, 30, 45, 60, 75, 90]
    ll = np.zeros_like(incs, dtype=float)
    for i, inc in enumerate(incs):
        map.inc = inc
        ll[i] = map.lnlike(**kwargs)

    # Verify that we get the correct inclination
    assert incs[np.argmax(ll)] == 60
    assert np.allclose(ll[np.argmax(ll)], 974.6716006920597)  # benchmarked
