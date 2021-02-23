# -*- coding: utf-8 -*-
"""Test adding spots to the map."""
import starry
import numpy as np
from scipy.ndimage import gaussian_filter1d


def test_normalization():
    map = starry.Map(30)
    contrast = 0.75
    baseline = 1 / np.pi  # the default starry intensity baseline
    map.spot(contrast=contrast, radius=20, lat=0, lon=0)
    actual_contrast = (baseline - map.intensity(lat=0, lon=0)) / baseline
    assert np.abs(contrast - actual_contrast) < 0.01


def test_profile():
    # Setttings (hand-tuned)
    radius = 20
    spot_fac = 300
    spot_smoothing = 0.075
    gaussian_smoothing = 25
    med_tol = 0.01
    max_tol = 0.05

    # Get the starry profile
    map = starry.Map(30)
    map.spot(
        contrast=1,
        radius=radius,
        lat=0,
        lon=0,
        spot_fac=spot_fac,
        spot_smoothing=spot_smoothing,
    )
    lon = np.linspace(-90, 90, 1000)
    I = np.pi * map.intensity(lon=lon)

    # The exact sigmoid we're expanding, with a Gaussian smoothing filter
    I0 = 1.0 / (1.0 + np.exp(-spot_fac * (np.abs(lon) - radius)))
    I0 = gaussian_filter1d(I0, gaussian_smoothing)

    # Check that they agree
    assert np.median(np.abs(I - I0)) < med_tol
    assert np.max(np.abs(I - I0)) < max_tol


def test_latlon():
    lat0 = 30
    lon0 = 80
    map = starry.Map(20)
    map.spot(lat=lat0, lon=lon0)
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
    The intensities at the spot centers should be the same.

    """

    map = starry.Map(ydeg=10)
    map.spot(lon=0)
    map.spot(lon=90)
    return np.isclose(map.intensity(lon=0), map.intensity(lon=90))
