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

# Gradient of flux when xo = 0 is NAN
@pytest.mark.xfail
def test_gradient_xo_zero():
    map = starry.Map(2)
    assert np.isfinite(map.flux(xo=0, ro=0.1, gradient=True)[1]["xo"])