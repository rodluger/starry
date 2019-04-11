"""
Test current bugs/issues in starry.

"""
import starry
import pytest
import numpy as np


# This was at one point causing segfaults
# Seems ok now...
def test_memleak():
    map = starry.Map(0, 2)
    flux, grad = map.flux(b=0.5, ro=0.1, gradient=True)
    dfdu = np.array(grad['u'])
    for i in range(100):
        flux, grad = map.flux(b=0.5, ro=0.1, gradient=True)
        assert np.allclose(grad['u'], dfdu)


# Bug when source = [0, 0, 1]!
# Total flux should be nonzero
@pytest.mark.xfail
def test_source_noon():
    map = starry.Map(2, reflected=True)
    assert np.nansum(map.render(source=[0, 0, 1])) != 0

# Not sure what's happening here.
def test_rv_hysteresis():
    map = starry.DopplerMap()
    map.veq = 1
    xo = np.linspace(-1.5, 1.5, 10)
    f1 = map.rv(xo=xo, ro=0.1)
    f2, _ = map.rv(xo=xo, ro=0.1, gradient=True)
    assert np.allclose(f1, f2)

# TODO: Gradient of RV when xo = 0 is NAN