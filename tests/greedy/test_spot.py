# -*- coding: utf-8 -*-
"""Test adding spots to the map."""
import starry
import pytest
import theano
import numpy as np


def test_normalization():
    map = starry.Map(20)
    amp = -np.random.random()
    map.add_spot(amp)
    assert map.amp == 1 + amp


def test_gaussianity():
    # Compute the expansion of the intensity with starry
    amp = 1.0
    sigma = 0.1
    map = starry.Map(30)
    map.add_spot(amp, sigma=sigma)
    lon = np.linspace(-90, 90, 100)
    I = (map.intensity(lon=lon) - 1 / np.pi) / (
        map.intensity(lon=0) - 1 / np.pi
    )

    # This is the actual gaussian spot intensity
    coslon = np.cos(lon * np.pi / 180)
    I_gaussian = np.exp(-((coslon - 1) ** 2) / (2 * sigma ** 2))

    # Compare
    assert np.allclose(I, I_gaussian, atol=1e-5)


def test_latlon():
    lat0 = 30
    lon0 = 80
    map = starry.Map(20)
    map.add_spot(-0.5, lat=lat0, lon=lon0)
    lat = np.arange(-90, 90, dtype="float64")
    lon = np.arange(-180, 180, dtype="float64")
    lat, lon = np.meshgrid(lat, lon)
    lat = lat.reshape(-1)
    lon = lon.reshape(-1)
    I = map.intensity(lat=lat, lon=lon)
    imin = np.argmin(I)
    assert lat[imin] == lat0
    assert lon[imin] == lon0
