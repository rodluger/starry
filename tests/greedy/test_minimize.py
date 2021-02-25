# -*- coding: utf-8 -*-
"""Test map minimum finding."""
import starry
import numpy as np


def test_minimize():
    # Load the Earth
    map = starry.Map(ydeg=10)
    map.load("earth")

    # Render it on a lat-lon grid
    res = 300
    image = map.render(projection="rect", res=res)

    # Find the minimum numerically
    lon, lat = np.meshgrid(
        np.linspace(-180, 180, res), np.linspace(-90, 90, res)
    )
    i, j = np.unravel_index(image.argmin(), image.shape)
    lat = lat[i, j]
    lon = lon[i, j]
    val = image[i, j]
    lat_m, lon_m, val_m, info = map.minimize(
        oversample=2, ntries=2, return_info=True
    )

    # Check that we did better than the grid search
    assert val_m <= val


def test_bounded_minimize():
    # Create map with two dark spots
    map = starry.Map(15)
    map.spot(contrast=0.1, lat=20, lon=50.0, radius=20)
    map.spot(contrast=0.05, lat=-30, lon=-40, radius=20)

    # Render it on a lat-lon grid
    res = 300
    image = map.render(projection="rect", res=res)

    # Specify bounds in latitude/longitude
    bounds = ((-60, 0), (-60, 0))  # lat and lon

    # Find the minimum numerically
    lon, lat = np.meshgrid(
        np.linspace(-180, 180, res), np.linspace(-90, 90, res)
    )
    mask_lat = np.logical_and(lat > bounds[0][0], lat < bounds[0][1])
    mask_lon = np.logical_and(lon > bounds[1][0], lon < bounds[1][1])
    mask = np.logical_and(mask_lat, mask_lon)
    min_bounded = image[mask].min()
    i, j = np.unravel_index(
        np.argmin(np.abs(image - min_bounded)), image.shape
    )

    lat = lat[i, j]
    lon = lon[i, j]
    val = image[i, j]

    lat_m, lon_m, val_m = map.minimize(oversample=2, ntries=2, bounds=bounds)

    assert val_m <= val


def test_sturm():
    # Check that we can count the real
    # roots of a polynomial in the range [0, 1]
    # using Sturm's theorem
    np.random.seed(1)
    nroots = starry._c_ops.nroots
    for i in range(100):
        p = np.random.randn(10)
        np_roots = [
            r.real
            for r in np.roots(p)
            if r.imag == 0 and r.real >= 0 and r.real <= 1
        ]
        np_nroots = len(np_roots)
        assert nroots(p, 0, 1) == np_nroots


def test_limbdark_physical():
    # Test our routine on quadratic LD, where
    # the constraints are analytic (Kipping 2013)
    np.random.seed(0)

    def is_physical(u):
        return (u[1] + u[2] < 1) and (u[1] > 0) and (u[1] + 2 * u[2] > 0)

    map = starry.Map(udeg=2)
    for i in range(500):
        map[1:] = np.random.randn(2)
        assert map.limbdark_is_physical() == is_physical(map.u)
