"""
Current bugs/issues in starry.

"""

import starry2
import pytest


@pytest.mark.xfail
def test_hysteresis():
    """Something is wrong with the map degree here."""
    map = starry2.Map()
    map[0,0] = 1
    map[1] = 1
    map[1, 0] = 1
    try:
        f, g = map.flux(xo=[0.1, 0.2], ro=0.25, gradient=True)
    except RuntimeError:
        pass
    map[1, 0] = 0
    f, g = map.flux(xo=[0.1, 0.2], ro=0.25, gradient=True)
    return 0


@pytest.mark.xfail
def test_cel_convergence():
    """The elliptic integral doesn't converge in this case."""
    map = starry2.Map()
    map[0, 0] = 1
    map[:] = 1
    flux0 = map.flux(xo=0, ro=0.01)
    flux = map.flux(xo=1e-16, ro=0.01)
    assert np.isclose(flux, flux0)