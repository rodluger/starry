"""
Test current bugs/issues in starry.

"""
import starry
import pytest
import numpy as np

map = starry.Map(0, 2)

flux, grad = map.flux(b=0.5, ro=0.1, gradient=True)
dfdu = np.array(grad['u'])

for i in range(100):
    flux, grad = map.flux(b=0.5, ro=0.1, gradient=True)
    assert np.allclose(grad['u'], dfdu)