import starry
import numpy as np


def test_conv2d():
    """
    Test that our design matrix and conv2d implementations of the flux
    yield identical results.

    """
    map = starry.DopplerMap(ydeg=10, nt=3, nc=2, nw=199, veq=50000)
    map.load(["spot", "earth"], force_psd=True)
    flux1 = map.flux(mode="conv")
    flux2 = map.flux(mode="design")
    assert np.allclose(flux1, flux2)
