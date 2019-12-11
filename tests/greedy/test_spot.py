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


def test_two_spots():
    """
    Test the behavior when adding two identical spots in different locations.

    If `relative` is True, the intensities at the spot centers should be
    different, since the user is asking for the spot to change the *current*
    intensity or amplitude of the map by a fixed percentage. After the first
    spot is added, the intensity/amplitude of the map has changed, so the
    second spot will result in a different *absolute* change to the intensity.

    If `relative` is False, the intensities at the spot centers should be
    the same. The user is essentially asking for the spot to change the
    *original* intensity or amplitude of the map by a fixed percentage.
    """

    def same_intensity(intensity=None, amp=None, relative=True):
        map = starry.Map(ydeg=10)
        map.add_spot(
            intensity=intensity, amp=amp, sigma=0.05, lon=0, relative=relative
        )
        map.add_spot(
            intensity=intensity, amp=amp, sigma=0.05, lon=90, relative=relative
        )
        return np.isclose(map.intensity(lon=0), map.intensity(lon=90))

    assert same_intensity(amp=-0.01, relative=False)
    assert same_intensity(intensity=-0.1, relative=False)

    assert not same_intensity(amp=-0.01, relative=True)
    assert not same_intensity(intensity=-0.1, relative=True)
