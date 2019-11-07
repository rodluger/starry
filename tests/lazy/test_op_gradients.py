# -*- coding: utf-8 -*-
"""
Test the various Theano Ops and their gradients.

"""
import theano
from theano.tests.unittest_tools import verify_grad
import theano.tensor as tt
import numpy as np
import pytest
import starry


def test_sT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2)
    verify_grad(
        map.ops.sT,
        (np.linspace(0.01, 1.09, 30), 0.1),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_intensity(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2, udeg=2)
    np.random.seed(11)
    lat = 180 * (np.random.random(10) - 0.5)
    lon = 360 * (np.random.random(10) - 0.5)
    y = [1.0] + list(np.random.randn(8))
    u = [-1.0] + list(np.random.randn(2))
    f = [np.pi]
    verify_grad(
        map.ops.intensity,
        (lat, lon, y, u, f),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_tensordotRz(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2)
    theta = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]) * np.pi / 180.0

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(
        map.ops.tensordotRz,
        (M, theta),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Vector M
    M = np.ones((1, 9))
    verify_grad(
        map.ops.tensordotRz,
        (M, theta),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_dotR(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2)

    x = 1.0 / np.sqrt(3)
    y = 1.0 / np.sqrt(3)
    z = 1.0 / np.sqrt(3)
    theta = np.pi / 5

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(
        map.ops.dotR,
        (M, x, y, z, theta),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Vector M
    M = np.ones((1, 9))
    verify_grad(
        map.ops.dotR,
        (M, x, y, z, theta),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_F(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2, udeg=2, rv=True)
    np.random.seed(11)
    u = np.random.randn(3)
    u[0] = -1
    f = np.random.randn(16)
    verify_grad(
        map.ops.F, (u, f), abs_tol=abs_tol, rel_tol=rel_tol, eps=eps, n_tests=1
    )


def test_pT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2)
    map[1:, :] = 1
    x = np.array([0.13])
    y = np.array([0.25])
    z = np.sqrt(1 - x ** 2 - y ** 2)
    verify_grad(
        map.ops.pT,
        (x, y, z),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_flux(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2)
    theta = np.linspace(0, 30, 10)
    xo = np.linspace(-1.5, 1.5, len(theta))
    yo = np.ones_like(xo) * 0.3
    zo = 1.0 * np.ones_like(xo)
    ro = 0.1
    inc = 85.0 * np.pi / 180.0
    obl = 30.0 * np.pi / 180.0
    y = np.ones(9)
    u = [-1.0]
    f = [np.pi]
    alpha = 0.0

    func = lambda *args: tt.dot(map.ops.X(*args), y)

    # Just rotation
    verify_grad(
        func,
        (theta, xo, yo, zo, 0.0, inc, obl, u, f, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Just occultation
    verify_grad(
        func,
        (theta, xo / 3, yo, zo, ro, inc, obl, u, f, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Rotation + occultation
    verify_grad(
        func,
        (theta, xo, yo, zo, ro, inc, obl, u, f, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_rT_reflected(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2, reflected=True)
    bterm = np.linspace(-1, 1, 10)[1:-1]
    verify_grad(
        map.ops.rT,
        (bterm,),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_intensity_reflected(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2, udeg=2, reflected=True)
    np.random.seed(11)
    lat = 180 * (np.random.random(10) - 0.5)
    lon = 360 * (np.random.random(10) - 0.5)
    y = [1.0] + list(np.random.randn(8))
    u = [-1.0] + list(np.random.randn(2))
    f = [np.pi, 0.0, 0.0, 0.0]
    source = np.random.randn(10, 3)
    source /= np.sqrt(np.sum(source ** 2, axis=1)).reshape(-1, 1)
    xo = source[:, 0]
    yo = source[:, 1]
    zo = source[:, 2]
    verify_grad(
        map.ops.intensity,
        (lat, lon, y, u, f, xo, yo, zo),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_flux_reflected(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2, reflected=True)
    theta = np.linspace(0, 30, 10)
    xo = np.linspace(-1.5, 1.5, len(theta))
    yo = np.ones_like(xo) * 0.3
    zo = -1.0 * np.ones_like(xo)
    ro = 0.1
    inc = 85.0 * np.pi / 180.0
    obl = 30.0 * np.pi / 180.0
    y = np.ones(9)
    u = [-1.0]
    f = [np.pi, 0.0, 0.0, 0.0]
    alpha = 0.0

    func = lambda *args: tt.dot(map.ops.X(*args), y)

    # Just rotation
    verify_grad(
        func,
        (theta, xo, yo, zo, ro, inc, obl, u, f, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_flux_ylm_ld(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2, udeg=2)
    theta = np.linspace(0, 30, 10)
    xo = np.linspace(-1.5, 1.5, len(theta))
    yo = np.ones_like(xo) * 0.3
    zo = 1.0 * np.ones_like(xo)
    ro = 0.1
    inc = 85.0 * np.pi / 180.0
    obl = 30.0 * np.pi / 180.0
    y = np.ones(9)
    np.random.seed(14)
    u = [-1.0] + list(np.random.randn(2))
    f = [np.pi]
    alpha = 0.0

    func = lambda *args: tt.dot(map.ops.X(*args), y)

    # Just rotation
    verify_grad(
        func,
        (theta, xo, yo, zo, 0.0, inc, obl, u, f, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Just occultation
    verify_grad(
        func,
        (theta, xo / 3, yo, zo, ro, inc, obl, u, f, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Rotation + occultation
    verify_grad(
        func,
        (theta, xo, yo, zo, ro, inc, obl, u, f, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_flux_quad_ld(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(udeg=2)
    xo = np.linspace(-1.5, 1.5, 10)
    yo = np.ones_like(xo) * 0.3
    zo = 1.0 * np.ones_like(xo)
    ro = 0.1
    np.random.seed(14)
    u = np.array([-1.0] + list(np.random.randn(2)))
    func = lambda *args: map.ops.flux(*args)

    verify_grad(
        func,
        (xo, yo, zo, ro, u),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_rv(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2, rv=True)
    theta = np.linspace(0, 30, 10)
    xo = np.linspace(-1.5, 1.5, len(theta))
    yo = np.ones_like(xo) * 0.3
    zo = 1.0 * np.ones_like(xo)
    ro = 0.1
    inc = 85.0 * np.pi / 180.0
    obl = 30.0 * np.pi / 180.0
    veq = 0.5
    alpha = 0.3
    y = np.ones(9)
    u = [-1.0]

    # Just rotation
    verify_grad(
        map.ops.rv,
        (theta, xo, yo, zo, 0.0, inc, obl, y, u, veq, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Just occultation
    verify_grad(
        map.ops.rv,
        (theta, xo / 3, yo, zo, ro, inc, obl, y, u, veq, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Rotation + occultation
    verify_grad(
        map.ops.rv,
        (theta, xo, yo, zo, ro, inc, obl, y, u, veq, alpha),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )
