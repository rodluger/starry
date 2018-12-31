"""Test the stability of the calculations near singular points."""
import starry2
import numpy as np
import pytest


@pytest.mark.xfail
def test_cel_convergence():
    """The elliptic integral doesn't converge in this case."""
    map = starry2.Map()
    map[0, 0] = 1
    map[:] = 1
    flux0 = map.flux(xo=0, ro=0.01)
    flux = map.flux(xo=1e-16, ro=0.01)
    assert np.isclose(flux, flux0)


def test_small_b():
    """Test df/db near b = 0."""
    b = np.logspace(-18, 0, 100)
    ro = 0.1

    # Double precision
    map = starry2.Map(10)
    map[0, 0] = 1
    map[:] = 1
    _, grad = map.flux(xo=b, ro=ro, gradient=True)
    dfdb = grad['xo']
    
    # Multiprecision
    map = starry2.Map(10, multi=True)
    map[0, 0] = 1
    map[:] = 1
    _, grad = map.flux(xo=b, ro=ro, gradient=True)
    dfdb_multi = grad['xo']

    # Compare
    assert(np.max(np.abs(dfdb - dfdb_multi)) < 1.e-11)