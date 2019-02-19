"""Test the maximum likelihood map solver."""
import starry
import numpy as np
import pytest


def test_linear_model():
    lmax = 5
    nt = 100
    map = starry.Map(lmax)
    np.random.seed(43)
    map[:, :] = np.random.randn(map.N)
    theta = np.linspace(0, 30, nt)
    xo = np.linspace(-1.5, 1.5, nt)
    yo = np.linspace(-0.1, 0.3, nt)
    zo = 1
    ro = 0.1

    # Compute the flux the usual way
    flux = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, zo=zo)

    # Compute it via the `linear_model` framework
    A = map.linear_model(theta=theta, xo=xo, yo=yo, ro=ro, zo=zo)
    flux_linear = np.dot(A, map.y)

    assert np.allclose(flux, flux_linear)


def test_linear_model_gradients():
    lmax = 5
    nt = 100
    map = starry.Map(lmax)
    np.random.seed(43)
    map[:, :] = np.random.randn(map.N)
    theta = np.linspace(0, 30, nt)
    xo = np.linspace(-1.5, 1.5, nt)
    yo = np.linspace(-0.1, 0.3, nt)
    zo = 1
    ro = 0.1

    # DEBUG
    yo = -99

    # Compute the flux the usual way
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, zo=zo, gradient=True)

    # Compute it via the `linear_model` framework
    A, dA = map.linear_model(theta=theta, xo=xo, yo=yo, ro=ro, zo=zo, gradient=True)
    flux_linear = np.dot(A, map.y)
    dfdtheta = np.dot(dA['theta'], map.y)
    assert np.allclose(flux, flux_linear)
    assert np.allclose(dfdtheta, grad['theta'])


if __name__ == "__main__":
    test_linear_model_gradients()