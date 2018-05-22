"""Test high order limb darkening."""
from starry import LimbDarkenedMap
import numpy as np
from scipy.integrate import dblquad


def NumericalFlux(b, r, u):
    """Compute the flux by numerical integration of the surface integral."""
    # I'm only coding up a specific case here
    assert (b >= 0) and (r <= 1), "Invalid range."

    if b >= 1 + r:
        return 1

    # Get points of intersection
    if b > 1 - r:
        yi = (1. + b ** 2 - r ** 2) / (2. * b)
        xi = (1. / (2. * b)) * np.sqrt(4 * b ** 2 - (1 + b ** 2 - r ** 2) ** 2)
    else:
        yi = np.inf
        xi = r

    # Specific intensity map
    norm = np.pi * (1 - 2 * np.sum([u[l] / ((l + 1) * (l + 2))
                                    for l in range(1, len(u))]))

    def I(y, x):
        mu = np.sqrt(1 - x ** 2 - y ** 2)
        return (1 - np.sum([u[l] * (1 - mu) ** l
                            for l in range(1, len(u))])) / norm

    # Lower integration limit
    def y1(x):
        if yi <= b:
            # Lower occultor boundary
            return b - np.sqrt(r ** 2 - x ** 2)
        elif b <= 1 - r:
            # Lower occultor boundary
            return b - np.sqrt(r ** 2 - x ** 2)
        else:
            # Tricky: we need to do this in two parts
            return b - np.sqrt(r ** 2 - x ** 2)

    # Upper integration limit
    def y2(x):
        if yi <= b:
            # Upper occulted boundary
            return np.sqrt(1 - x ** 2)
        elif b <= 1 - r:
            # Upper occultor boundary
            return b + np.sqrt(r ** 2 - x ** 2)
        else:
            # Tricky: we need to do this in two parts
            return np.sqrt(1 - x ** 2)

    # Compute the total flux
    flux, _ = dblquad(I, -xi, xi, y1, y2, epsabs=1e-14, epsrel=1e-14)

    # Do we need to solve an additional integral?
    if not (yi <= b) and not (b <= 1 - r):

        def y1(x):
            return b - np.sqrt(r ** 2 - x ** 2)

        def y2(x):
            return b + np.sqrt(r ** 2 - x ** 2)

        additional_flux, _ = dblquad(I, -r, -xi, y1, y2,
                                     epsabs=1e-14, epsrel=1e-14)

        flux += 2 * additional_flux

    return 1 - flux


def test_transits():
    """Test transit light curve generation for 8th order limb darkening."""
    # Input params
    u = [1, 0.4, 0.26, 0.3, 0.5, -0.2, 0.5, -0.7, 0.3]
    npts = 25
    r = 0.1
    b = np.linspace(0, 1 + r + 0.1, npts)

    # Numerical flux
    nF = np.zeros_like(b)
    for i in range(npts):
        nF[i] = NumericalFlux(b[i], r, u)
    den = (1 - nF)
    den[den == 0] = 1e-10

    # Compute the starry flux
    map = LimbDarkenedMap(len(u) - 1)
    for l in range(1, len(u)):
        map[l] = u[l]
    sF = map.flux(xo=b, yo=0, ro=r)

    # Compute the error, check that it's better than 1 ppb
    error = np.max((np.abs(nF - sF) / den)[np.where(nF < 1)])
    assert error < 1e-9


if __name__ == "__main__":
    test_transits()
