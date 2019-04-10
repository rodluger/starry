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


# Occultation flux is wrong when using filters
@pytest.mark.xfail
def test_filter_flux():
    map = starry.Map(1, fdeg=1)
    map.filter[1, 1] = 1
    f1 = np.array(map.flux(xo=0.1, ro=0.1))[0]

    # Note that the first term of the design matrix
    # is ZERO: this is the source of the problem!
    print(map.linear_flux_model(xo=0.1, ro=0.1))

    map = starry.Map(1)
    fbase = np.array(map.flux(xo=0.1, ro=0.1))[0]
    map[1, 1] = 1 / np.pi
    f2 = np.array(map.flux(xo=0.1, ro=0.1))[0] - fbase

    print(f1, f2)
    assert f1 == f2


if __name__ == "__main__":
    test_filter_flux()
