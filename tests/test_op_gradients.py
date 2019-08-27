# -*- coding: utf-8 -*-
"""
Test the various Theano Ops.

"""
import theano

theano.config.compute_test_value = "off"

from theano.tests.unittest_tools import verify_grad
import theano.tensor as tt
import starry
import numpy as np


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
    xpt = 0.5 * np.random.random(10)
    ypt = 0.5 * np.random.random(10)
    zpt = np.sqrt(1 - xpt ** 2 - ypt ** 2)
    y = [1.0] + list(np.random.randn(8))
    u = [-1.0] + list(np.random.randn(2))
    f = [np.pi]
    verify_grad(
        map.ops.intensity,
        (xpt, ypt, zpt, y, u, f),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_dotRz(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2)
    theta = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]) * np.pi / 180.0

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(
        map.ops.dotRz,
        (M, theta),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Vector M
    M = np.ones((1, 9))
    verify_grad(
        map.ops.dotRz,
        (M, theta),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_dotRxy(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2)
    inc = 85.0 * np.pi / 180.0
    obl = 30.0 * np.pi / 180.0

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(
        map.ops.dotRxy,
        (M, inc, obl),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Vector M
    M = np.ones((1, 9))
    verify_grad(
        map.ops.dotRxy,
        (M, inc, obl),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_dotRxyT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    theano.config.compute_test_value = "off"
    map = starry.Map(ydeg=2)
    inc = 85.0 * np.pi / 180.0
    obl = 30.0 * np.pi / 180.0

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(
        map.ops.dotRxyT,
        (M, inc, obl),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Vector M
    M = np.ones((1, 9))
    verify_grad(
        map.ops.dotRxyT,
        (M, inc, obl),
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

    func = lambda *args: tt.dot(map.ops.X(*args), y)

    # Just rotation
    verify_grad(
        func,
        (theta, xo, yo, zo, 0.0, inc, obl, u, f),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Just occultation
    verify_grad(
        func,
        (theta, xo / 3, yo, zo, ro, inc, obl, u, f),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Rotation + occultation
    verify_grad(
        func,
        (theta, xo, yo, zo, ro, inc, obl, u, f),
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
    xpt = 0.5 * np.random.random(10)
    ypt = 0.5 * np.random.random(10)
    zpt = np.sqrt(1 - xpt ** 2 - ypt ** 2)
    y = [1.0] + list(np.random.randn(8))
    u = [-1.0] + list(np.random.randn(2))
    f = [np.pi]
    source = source = np.random.randn(10, 3)
    source /= np.sqrt(np.sum(source ** 2, axis=1)).reshape(-1, 1)
    verify_grad(
        map.ops.intensity,
        (xpt, ypt, zpt, y, u, f, source),
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
    f = [np.pi]
    source = np.random.randn(len(theta), 3)
    source /= np.sqrt(np.sum(source ** 2, axis=1)).reshape(-1, 1)

    func = lambda *args: tt.dot(map.ops.X(*args), y)

    # Just rotation
    verify_grad(
        func,
        (theta, xo, yo, zo, ro, inc, obl, u, f, source),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )


def test_flux_ld(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
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

    func = lambda *args: tt.dot(map.ops.X(*args), y)

    # Just rotation
    verify_grad(
        func,
        (theta, xo, yo, zo, 0.0, inc, obl, u, f),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Just occultation
    verify_grad(
        func,
        (theta, xo / 3, yo, zo, ro, inc, obl, u, f),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eps=eps,
        n_tests=1,
    )

    # Rotation + occultation
    verify_grad(
        func,
        (theta, xo, yo, zo, ro, inc, obl, u, f),
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


if __name__ == "__main__":
    """
    theano.config.compute_test_value = "raise"
    map = starry.Map(ydeg=2)
    verify_grad(map.ops.sT, (np.linspace(0.01, 1.09, 30), 0.1), n_tests=1)
    """
    pass

