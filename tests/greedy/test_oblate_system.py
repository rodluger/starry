import starry
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize("ydeg,nw", [[0, None], [0, 10], [1, None], [1, 10]])
def test_system(ydeg, nw):

    # Oblate map
    map = starry.Map(udeg=2, ydeg=ydeg, oblate=True, nw=nw)
    map[1] = 0.5
    map[2] = 0.25
    map.omega = 0.5
    map.beta = 1.23
    map.tpole = 8000
    map.f = 1 - 2 / (map.omega ** 2 + 2)
    map.obl = 30

    # Compute system flux
    star = starry.Primary(map, r=1.5)
    planet = starry.Secondary(starry.Map(amp=0, nw=nw), porb=1.0, r=0.1, m=0)
    sys = starry.System(star, planet)
    t = np.linspace(-0.1, 0.1, 1000)
    flux_sys = sys.flux(t, integrated=True)

    # Compute map flux manually
    x, y, z = sys.position(t)
    xo = x[1] / star._r
    yo = y[1] / star._r
    flux_map = map.flux(xo=xo, yo=yo, ro=planet._r / star._r, integrated=True)

    # Check that they agree
    assert np.allclose(flux_map, flux_sys)
