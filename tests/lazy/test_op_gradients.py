# -*- coding: utf-8 -*-
"""
Test the various Theano Ops and their gradients.

"""
import theano
from starry.compat import change_flags
import theano.tensor as tt
import numpy as np
import pytest
import starry


def test_sT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2)
        tt.verify_grad(
            map.ops.sT,
            (np.linspace(0.01, 1.09, 30), 0.1),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_intensity(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2, udeg=2)
        np.random.seed(11)
        lat = 180 * (np.random.random(10) - 0.5)
        lon = 360 * (np.random.random(10) - 0.5)
        y = [1.0] + list(np.random.randn(8))
        u = [-1.0] + list(np.random.randn(2))
        f = [np.pi]
        theta = 0.0

        def intensity(lat, lon, y, u, f, theta):
            return map.ops.intensity(lat, lon, y, u, f, theta, np.array(True))

        tt.verify_grad(
            intensity,
            (lat, lon, y, u, f, theta),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_tensordotRz(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2)
        theta = (
            np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]) * np.pi / 180.0
        )

        # Matrix M
        M = np.ones((7, 9))
        tt.verify_grad(
            map.ops.tensordotRz,
            (M, theta),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )

        # Vector M
        M = np.ones((1, 9))
        tt.verify_grad(
            map.ops.tensordotRz,
            (M, theta),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_dotR(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2)

        x = 1.0 / np.sqrt(3)
        y = 1.0 / np.sqrt(3)
        z = 1.0 / np.sqrt(3)
        theta = np.pi / 5

        # Matrix M
        M = np.ones((7, 9))
        tt.verify_grad(
            map.ops.dotR,
            (M, x, y, z, theta),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )

        # Vector M
        M = np.ones((1, 9))
        tt.verify_grad(
            map.ops.dotR,
            (M, x, y, z, theta),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_F(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2, udeg=2, rv=True)
        np.random.seed(11)
        u = np.random.randn(3)
        u[0] = -1
        f = np.random.randn(16)
        tt.verify_grad(
            map.ops.F,
            (u, f),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_pT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2)
        map[1:, :] = 1
        x = np.array([0.13])
        y = np.array([0.25])
        z = np.sqrt(1 - x ** 2 - y ** 2)
        tt.verify_grad(
            map.ops.pT,
            (x, y, z),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_flux(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
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
        tt.verify_grad(
            func,
            (theta, xo, yo, zo, 0.0, inc, obl, u, f),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )

        # Just occultation
        tt.verify_grad(
            func,
            (theta, xo / 3, yo, zo, ro, inc, obl, u, f),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )

        # Rotation + occultation
        tt.verify_grad(
            func,
            (theta, xo, yo, zo, ro, inc, obl, u, f),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_rT_reflected(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2, reflected=True)
        bterm = np.linspace(-1, 1, 10)[1:-1]
        sigr = 30 * np.pi / 180
        tt.verify_grad(
            map.ops.rT,
            (bterm, sigr),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


"""
# TODO: Implement the gradient of the OrenNayarOp.
def test_intensity_reflected(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2, udeg=2, reflected=True)
        np.random.seed(11)
        lat = 180 * (np.random.random(10) - 0.5)
        lon = 360 * (np.random.random(10) - 0.5)
        y = [1.0] + list(np.random.randn(8))
        u = [-1.0] + list(np.random.randn(2))
        f = [np.pi]
        source = np.random.randn(10, 3)
        source /= np.sqrt(np.sum(source ** 2, axis=1)).reshape(-1, 1)
        xs = source[:, 0]
        ys = source[:, 1]
        zs = source[:, 2]
        Rs = 1.0
        theta = 0.0
        sigr = 30 * np.pi / 180

        def intensity(lat, lon, y, u, f, xs, ys, zs, Rs, theta, sigr):
            return map.ops.intensity(
                lat,
                lon,
                y,
                u,
                f,
                xs,
                ys,
                zs,
                Rs,
                theta,
                np.array(False),
                sigr,
                np.array(False),
                np.array(True),
            )

        tt.verify_grad(
            intensity,
            (lat, lon, y, u, f, xs, ys, zs, Rs, theta, sigr),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1, rng=np.random,
        )
"""


def test_flux_reflected(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2, reflected=True)
        theta = np.linspace(0, 30, 10)
        xs = np.linspace(-1.5, 1.5, len(theta))
        ys = np.ones_like(xs) * 0.3
        zs = -1.0 * np.ones_like(xs)
        ro = 0.0
        inc = 85.0 * np.pi / 180.0
        obl = 30.0 * np.pi / 180.0
        y = np.ones(9)
        u = [-1.0]
        f = [np.pi]
        Rs = 1.0
        sigr = 30 * np.pi / 180

        def func(theta, xs, ys, zs, Rs, ro, inc, obl, u, f):
            return tt.dot(
                map.ops.X(
                    theta, xs, ys, zs, Rs, xs, ys, zs, ro, inc, obl, u, f, sigr
                ),
                y,
            )

        # Just rotation
        tt.verify_grad(
            func,
            (theta, xs, ys, zs, Rs, ro, inc, obl, u, f),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_flux_ylm_ld(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
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
        tt.verify_grad(
            func,
            (theta, xo, yo, zo, 0.0, inc, obl, u, f),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )

        # Just occultation
        tt.verify_grad(
            func,
            (theta, xo / 3, yo, zo, ro, inc, obl, u, f),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )

        # Rotation + occultation
        tt.verify_grad(
            func,
            (theta, xo, yo, zo, ro, inc, obl, u, f),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_flux_quad_ld(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(udeg=2)
        xo = np.linspace(-1.5, 1.5, 10)
        yo = np.ones_like(xo) * 0.3
        zo = 1.0 * np.ones_like(xo)
        ro = 0.1
        np.random.seed(14)
        u = np.array([-1.0] + list(np.random.randn(2)))
        func = lambda *args: map.ops.flux(*args)

        tt.verify_grad(
            func,
            (xo, yo, zo, ro, u),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_rv(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
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
        tt.verify_grad(
            map.ops.rv,
            (theta, xo, yo, zo, 0.0, inc, obl, y, u, veq, alpha),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )

        # Just occultation
        tt.verify_grad(
            map.ops.rv,
            (theta, xo / 3, yo, zo, ro, inc, obl, y, u, veq, alpha),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )

        # Rotation + occultation
        tt.verify_grad(
            map.ops.rv,
            (theta, xo, yo, zo, ro, inc, obl, y, u, veq, alpha),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_spot(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=5)
        amp = [-0.01]
        sigma = 0.1
        lat = 30 * np.pi / 180
        lon = 45 * np.pi / 180
        tt.verify_grad(
            map.ops.spotYlm,
            (amp, sigma, lat, lon),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_spot_spectral(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=5, nw=2)
        amp = [-0.01, -0.02]
        sigma = 0.1
        lat = 30 * np.pi / 180
        lon = 45 * np.pi / 180
        tt.verify_grad(
            map.ops.spotYlm,
            (amp, sigma, lat, lon),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )


def test_sT_reflected(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        map = starry.Map(ydeg=2, reflected=True)
        b = np.array([0.5])
        theta = np.array([0.5])
        bo = np.array([0.75])
        ro = 0.5
        sigr = 30 * np.pi / 180
        tt.verify_grad(
            map.ops.sT,
            (b, theta, bo, ro, sigr),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            n_tests=1,
            rng=np.random,
        )
