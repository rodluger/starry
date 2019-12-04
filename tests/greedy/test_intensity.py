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


def test_design_matrix():
    map[1:, :] = [0.1, 0.2, 0.3]
    lat = [0, 30]
    lon = [45, 90]
    P = map.intensity_design_matrix(lat=lat, lon=lon)
    assert np.allclose(map.intensity(lat=lat, lon=lon), P.dot(map.y))


def test_limb_darkened():
    map = starry.Map(udeg=2)
    map[1] = 0.5
    map[2] = 0.25

    assert map.intensity(mu=0) == 1 - map[1] - map[2]  # limb
    assert map.intensity(mu=1) == 1.0  # center

    assert map.intensity(x=1, y=0) == 1 - map[1] - map[2]  # limb
    assert map.intensity(x=0, y=0) == 1.0  # center
