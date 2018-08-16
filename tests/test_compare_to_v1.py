"""Compare to the previous version of starry."""
import starry
import starry2
import numpy as np


def test_transit():
    """Compare transit evaluation to the previous version of starry."""
    npts = 100
    theta = np.linspace(0, 30, npts)
    xo = np.linspace(-1.5, 1.5, npts)
    yo = np.linspace(-0.3, 0.7, npts)
    ro = 0.1
    axis = [1, 2, 3]

    # Double precision
    map = starry.Map(5)
    map[:] = 1
    flux = map.flux(axis=axis, theta=theta, xo=xo, yo=yo, ro=ro)

    map2 = starry2.Map(5)
    map2[:, :] = 1
    map2.axis = axis
    flux2 = map2.flux(theta=theta, xo=xo, yo=yo, ro=ro)

    assert np.allclose(flux, flux2)

    # Multi precision
    map = starry.multi.Map(5)
    map[:] = 1
    flux = map.flux(axis=axis, theta=theta, xo=xo, yo=yo, ro=ro)

    map2 = starry2.multi.Map(5)
    map2[:, :] = 1
    map2.axis = axis
    flux2 = map2.flux(theta=theta, xo=xo, yo=yo, ro=ro)

    assert np.allclose(flux, flux2)


def test_gradient():
    """Compare transit evaluation w/ grad to the previous version of starry."""
    npts = 300
    theta = np.linspace(1, 359, npts)
    xo = np.linspace(-1.5, 1.5, npts)
    yo = np.linspace(-0.3, 0.7, npts)
    ro = 0.1
    axis = [1, 2, 3]

    # Double precision
    map = starry.grad.Map(5)
    map[:] = 1
    flux = map.flux(axis=axis, theta=theta, xo=xo, yo=yo, ro=ro)
    grad = map.gradient

    map2 = starry2.Map(5)
    map2[:, :] = 1
    map2.axis = axis
    flux2, grad2 = map2.flux(theta=theta, xo=xo, yo=yo, ro=ro, gradient=True)

    # Check the flux is close
    assert np.allclose(flux, flux2)

    # Check the gradients are close
    for key in grad.keys():
        assert(np.allclose(grad[key], grad2[key]))


if __name__ == "__main__":
    test_transit()
    test_gradient()
