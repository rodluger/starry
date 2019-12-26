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

    lats = [0.0001, 0.0, 30.0, 0.0]
    lons = [30.0, 30.0, 0.0, 0.0]

    for lat, lon in zip(lats, lons):
        with change_flags(compute_test_value="off"):
            map = starry.Map(ydeg=2, udeg=2)
            np.random.seed(11)
            y = [1.0] + list(np.random.randn(8))
            u = [-1.0] + list(np.random.randn(2))
            f = [np.pi]
            wta = 0.0

            def intensity(lat, lon, y, u, f, wta):
                return map.ops.intensity(
                    lat, lon, y, u, f, wta, np.array(True)
                )

            verify_grad(
                intensity,
                (lat, lon, y, u, f, wta),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                eps=eps,
                n_tests=1,
            )
