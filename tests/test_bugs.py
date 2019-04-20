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

# TODO: Limb darkening should be disabled for projection = 'rect'