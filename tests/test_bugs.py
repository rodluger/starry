"""
Test current bugs/issues in starry.

"""
import starry
import pytest
import numpy as np


# Gradient of flux when xo = 0 is NAN
@pytest.mark.xfail
def test_gradient_xo_zero():
    map = starry.Map(2)
    assert np.isfinite(map.flux(xo=0, ro=0.1, gradient=True)[1]["xo"])

@pytest.mark.xfail
def test_xo_yo_derivs():
    # There may be weird stuff going on in grad['xo'] when the occultor
    # touches the limb of the occulted...
    raise Exception("")

# TODO: If veq < 0, inc < -180, etc., don't throw error in theano Op;
#       instead, return 0 prob!