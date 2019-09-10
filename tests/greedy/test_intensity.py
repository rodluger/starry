# -*- coding: utf-8 -*-
"""Test the map intensity evaluation."""
import starry
import numpy as np
import pytest

# Setup
map = starry.Map(ydeg=1)
params = [
    [[0, 0, 0], 0, 0, 1.0 / np.pi],
    [[1, 0, 0], 0, np.linspace(-180, 180, 10), 1.0 / np.pi],
    [[0, 1, 0], 0, 0, 1.0 / np.pi + np.sqrt(3) / np.pi],
    [[0, 0, 1], np.linspace(-90, 90, 10), 0, 1.0 / np.pi],
    [
        [1, 0, 0],
        30,
        np.linspace(-180, 180, 10),
        1.0 / np.pi + np.sqrt(3) / (2.0 * np.pi),
    ],
]


@pytest.mark.parametrize("y,lat,lon,I", params)
def test_intensity(y, lat, lon, I):
    map[1:, :] = y
    assert np.allclose(map.intensity(lat=lat, lon=lon), I)
