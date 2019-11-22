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
    lat_m, lon_m, val_m = map.minimize()

    # The location and value should agree within 5 percent
    assert np.allclose([lat_m, lon_m], [lat, lon], atol=0, rtol=0.05)
    assert np.allclose(val_m, val, atol=0, rtol=0.05)
