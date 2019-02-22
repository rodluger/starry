"""Test the linear model method."""
import starry
import numpy as np
import pytest


def test_linear_model():
    lmax = 5
    npts = 100
    map = starry.Map(lmax)
    np.random.seed(43)
    map[:, :] = np.random.randn(map.N)
    theta = np.linspace(0, 30, npts)
    xo = np.linspace(-1.5, 1.5, npts)
    yo = np.linspace(-0.1, 0.3, npts)
    zo = 1
    ro = 0.1

    # Compute the flux the usual way
    flux = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, zo=zo)

    # Compute it via the `linear_model` framework
    A = map.linear_model(theta=theta, xo=xo, yo=yo, ro=ro, zo=zo)
    flux_linear = np.dot(A, map.y)

    assert np.allclose(flux, flux_linear)


def test_linear_model_temporal():
    lmax = 1
    npts = 10
    nt = 2
    map = starry.Map(lmax, nt=nt)
    np.random.seed(43)
    map[:, :] = np.random.randn(map.N, nt)
    theta = np.linspace(0, 30, npts)
    xo = np.linspace(-1.5, 1.5, npts)
    yo = np.linspace(-0.1, 0.3, npts)
    zo = 1
    ro = 0.5
    t = np.linspace(0.0, 1.0, npts)

    # Compute the flux the usual way
    flux = np.array(map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, zo=zo))

    # Compute it via the `linear_model` framework
    A = np.array(map.linear_model(t=t, theta=theta, xo=xo, yo=yo, ro=ro, zo=zo))
    y = map.y.transpose().reshape(-1)
    flux_linear = np.dot(A, y)
    assert np.allclose(flux, flux_linear)


def test_linear_model_gradients():
    lmax = 5
    npts = 100
    map = starry.Map(lmax)
    np.random.seed(43)
    map[:, :] = np.random.randn(map.N)
    theta = np.linspace(0, 30, npts)
    xo = np.linspace(-1.5, 1.5, npts)
    yo = np.linspace(-0.1, 0.3, npts)
    zo = 1
    ro = 0.1

    # Compute the flux the usual way
    flux, grad = map.flux(theta=theta, xo=xo, yo=yo, ro=ro, zo=zo, gradient=True)

    # Compute it via the `linear_model` framework
    A, dA = map.linear_model(theta=theta, xo=xo, yo=yo, ro=ro, zo=zo, gradient=True)
    flux_linear = np.dot(A, map.y)
    dfdtheta = np.dot(dA['theta'], map.y)
    dfdxo = np.dot(dA['xo'], map.y)
    dfdyo = np.dot(dA['yo'], map.y)
    dfdro = np.dot(dA['ro'], map.y)

    assert np.allclose(flux, flux_linear)
    assert np.allclose(dfdtheta, grad['theta'])
    assert np.allclose(dfdxo, grad['xo'])
    assert np.allclose(dfdyo, grad['yo'])
    assert np.allclose(dfdro, grad['ro'])


def test_linear_model_temporal_gradients():
    lmax = 5
    npts = 100
    nt = 2
    map = starry.Map(lmax, nt=nt)
    np.random.seed(43)
    map[:, :] = np.random.randn(map.N, nt)
    theta = np.linspace(0, 30, npts)
    xo = np.linspace(-1.5, 1.5, npts)
    yo = np.linspace(-0.1, 0.3, npts)
    zo = 1
    ro = 0.1
    t = np.linspace(0.0, 1.0, npts)

    # Compute the flux the usual way
    flux, grad = map.flux(t=t, theta=theta, xo=xo, yo=yo, ro=ro, zo=zo, gradient=True)

    # Compute it via the `linear_model` framework
    A, dA = map.linear_model(t=t, theta=theta, xo=xo, yo=yo, ro=ro, zo=zo, gradient=True)
    y = map.y.transpose().reshape(-1)
    flux_linear = np.dot(A, y)
    dfdt = np.dot(dA['t'], y)
    dfdtheta = np.dot(dA['theta'], y)
    dfdxo = np.dot(dA['xo'], y)
    dfdyo = np.dot(dA['yo'], y)
    dfdro = np.dot(dA['ro'], y)

    assert np.allclose(dfdt, grad['t'])
    assert np.allclose(dfdtheta, grad['theta'])
    assert np.allclose(dfdxo, grad['xo'])
    assert np.allclose(dfdyo, grad['yo'])
    assert np.allclose(dfdro, grad['ro'])


test_linear_model_temporal_gradients()