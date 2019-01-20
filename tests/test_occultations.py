"""Test the occultation flux."""
from starry2 import Map
import numpy as np
from scipy.integrate import dblquad


def NumericalFlux(map, xo=0.0, yo=0.5, ro=0.1, epsabs=1e-8, epsrel=1e-8):
    """Compute the flux by numerical integration of the surface integral."""
    # Get the intensity distribution
    I = lambda y, x: map(x=x, y=y)

    # I'm only coding up a specific case here!
    assert xo == 0, "The occultor must be along the y axis because I'm lazy."
    b = yo
    r = ro
    assert (b >= 0) and (r <= 1), "Invalid range."

    # Total flux
    total, _ = dblquad(I, -1, 1, lambda x: -np.sqrt(1 - x ** 2),
                       lambda x: np.sqrt(1 - x ** 2),
                       epsabs=epsabs, epsrel=epsrel)
    if b >= 1 + r:
        return total

    # Get points of intersection
    if b > 1 - r:
        yi = (1. + b ** 2 - r ** 2) / (2. * b)
        xi = (1. / (2. * b)) * np.sqrt(4 * b ** 2 - (1 + b ** 2 - r ** 2) ** 2)
    else:
        yi = np.inf
        xi = r

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
    flux, _ = dblquad(I, -xi, xi, y1, y2, epsabs=epsabs, epsrel=epsrel)

    # Do we need to solve additional integrals?
    if not (yi <= b) and not (b <= 1 - r):

        def y1(x):
            return b - np.sqrt(r ** 2 - x ** 2)

        def y2(x):
            return b + np.sqrt(r ** 2 - x ** 2)

        additional_flux1, _ = dblquad(I, -r, -xi, y1, y2,
                                      epsabs=epsabs, epsrel=epsrel)
        additional_flux2, _ = dblquad(I, xi, r, y1, y2,
                                      epsabs=epsabs, epsrel=epsrel)

        flux += additional_flux1 + additional_flux2

    return total - flux


if __name__ == "__main__":
    # TODO
    lmax = 3
    map = Map(lmax)
    map.random(np.ones(lmax), seed=12)
    yo = 0.5
    ro = 0.1
    print(NumericalFlux(map, yo=yo, ro=ro))