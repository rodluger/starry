# -*- coding: utf-8 -*-
"""
Test multi-wavelength maps.

"""
import starry
import pytest
import itertools

# Params combinations
ydeg = [2]
udeg = [0]
nw = [5]
rv = [False, True]
reflected = [False, True]
params = itertools.product(ydeg, udeg, nw, rv, reflected)

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
        assert map.flux().eval().shape == (1, map.nw)

    def test_flux_vector(self, map):
        assert map.flux(xo=[0, 1, 2]).eval().shape == (3, map.nw)

    def test_intensity(self, map):
        if map.reflected:
            assert map.intensity().eval().shape == (1, map.nw, 1)
        else:
            assert map.intensity().eval().shape == (1, map.nw)

    def test_intensity_vector(self, map):
        if map.reflected:
            assert map.intensity(lat=[0, 30, 60]).eval().shape == (
                3,
                map.nw,
                1,
            )
        else:
            assert map.intensity(lat=[0, 30, 60]).eval().shape == (3, map.nw)

    def test_intensity_matrix(self, map):
        if map.reflected:
            assert map.intensity(lat=[0, 30, 60], xs=[1, 2]).eval().shape == (
                3,
                map.nw,
                2,
            )
        else:
            pass

    def test_rv(self, map):
        if hasattr(map, "rv"):
            assert map.rv().eval().shape == (1, map.nw)
        else:
            pass

    def test_rv_vector(self, map):
        if hasattr(map, "rv"):
            assert map.rv(xo=[0, 1, 2]).eval().shape == (3, map.nw)
        else:
            pass

    def test_render(self, map):
        res = 100
        map.render(res=res).eval().shape == (map.nw, res, res)
