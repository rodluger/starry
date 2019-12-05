# -*- coding: utf-8 -*-
"""
Test multi-wavelength maps.

"""
import starry
import pytest
import itertools
import numpy as np


# Params combinations
ydeg = [2]
udeg = [0]
nw = [5]
rv = [False, True]
reflected = [False, True]
params = list(itertools.product(ydeg, udeg, nw, rv, reflected))

# Reflected light + radial velocity is not implemented
params = [p for p in params if not (p[3] and p[4])]


@pytest.fixture(scope="class", params=params)
def map(request):
    ydeg, udeg, nw, rv, reflected = request.param
    map = starry.Map(ydeg=ydeg, udeg=udeg, nw=nw, reflected=reflected, rv=rv)
    map.reflected = reflected
    return map


class TestShapes:
    """Test the flux and intensity return value shapes."""

    def test_flux(self, map):
        assert map.flux().shape == (1, map.nw)

    def test_flux_vector(self, map):
        assert map.flux(xo=[0, 1, 2]).shape == (3, map.nw)

    def test_intensity(self, map):
        if map.reflected:
            assert map.intensity().shape == (1, map.nw, 1)
        else:
            assert map.intensity().shape == (1, map.nw)

    def test_intensity_vector(self, map):
        if map.reflected:
            assert map.intensity(lat=[0, 30, 60]).shape == (3, map.nw, 1)
        else:
            assert map.intensity(lat=[0, 30, 60]).shape == (3, map.nw)

    def test_intensity_matrix(self, map):
        if map.reflected:
            assert map.intensity(lat=[0, 30, 60], xs=[1, 2]).shape == (
                3,
                map.nw,
                2,
            )
        else:
            pass

    def test_rv(self, map):
        if hasattr(map, "rv"):
            assert map.rv().shape == (1, map.nw)
        else:
            pass

    def test_rv_vector(self, map):
        if hasattr(map, "rv"):
            assert map.rv(xo=[0, 1, 2]).shape == (3, map.nw)
        else:
            pass

    def test_render(self, map):
        res = 100
        map.render(res=res).shape == (map.nw, res, res)


def test_amplitude():
    """Test the amplitude attribute of a multi-wavelength map."""
    map = starry.Map(ydeg=1, nw=5)
    assert np.allclose(map.amp, np.ones(5))
    map.amp = 10.0
    assert np.allclose(map.amp, 10.0 * np.ones(5))
