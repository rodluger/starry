import starry
import theano
from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags
import pytest
import numpy as np


def test_zero_lat_lon(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    """
    The gradient of the intensity is NaN when either the
    latitude or the longitude is zero. There's a div by zero in
    the Op that we need to work around.

    """

    lats = [0.0001, 0.0, np.pi / 6, 0.0]
    lons = [np.pi / 6, np.pi / 6, 0.0, 0.0]

    for lat, lon in zip(lats, lons):
        with change_flags(compute_test_value="off"):
            map = starry.Map(ydeg=2, udeg=2)
            np.random.seed(11)
            y = np.array([1.0] + list(np.random.randn(8)))
            u = np.array([-1.0] + list(np.random.randn(2)))
            f = np.array([np.pi])
            theta = np.array(0.0)
            alpha = np.array(0.1)
            tau = np.array(0.5)
            delta = np.array(0.0)
            lat = np.array(lat)
            lon = np.array(lon)

            def intensity(lat, lon, y, u, f, theta, alpha, tau, delta):
                return map.ops.intensity(
                    lat, lon, y, u, f, theta, alpha, tau, delta, np.array(True)
                )

            verify_grad(
                intensity,
                (lat, lon, y, u, f, theta, alpha, tau, delta),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                eps=eps,
                n_tests=1,
            )
