"""
Test current bugs/issues in starry.

"""
import starry2
import pytest
import numpy as np
import subprocess


@pytest.mark.xfail
def test_hysteresis():
    """Something is wrong with the map degree here."""
    map = starry2.Map(4)
    map[:, :] = 1
    map.flux(theta=30.0)
    map[4:, :] = 0
    map[1] = 1
    map.flux(theta=30.0)


@pytest.mark.xfail
def test_cel_convergence():
    """The elliptic integral doesn't converge in this case."""
    map = starry2.Map()
    map[0, 0] = 1
    map[:] = 1
    flux0 = map.flux(xo=0, ro=0.01)
    flux = map.flux(xo=1e-16, ro=0.01)
    assert np.isclose(flux, flux0)


@pytest.mark.xfail
def test_small_b():
    """
    Test df/db near b = 0. 
    
    This test occasionally fails, probably due to a memory
    initialization issue...
    """
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