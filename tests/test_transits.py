"""Test transit light curve generation."""
from starry2.kepler import Primary, Secondary, System
import numpy as np
from scipy.integrate import dblquad


def NumericalFlux(b, r, u1, u2):
    """Compute the flux by numerical integration of the surface integral."""
    # I'm only coding up a specific case here
    assert (b >= 0) and (r <= 1), "Invalid range."

    # Total flux
    total = (np.pi / 6) * (6 - 2 * u1 - u2)
    if b >= 1 + r:
        return total

    # Get points of intersection
    if b > 1 - r:
        yi = (1. + b ** 2 - r ** 2) / (2. * b)
        xi = (1. / (2. * b)) * np.sqrt(4 * b ** 2 - (1 + b ** 2 - r ** 2) ** 2)
    else:
        yi = np.inf
        xi = r

    # Specific intensity map
    def I(y, x):
        mu = np.sqrt(1 - x ** 2 - y ** 2)
        return 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2

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

    return total - flux


def test_transits():
    """Test transit light curve generation."""
    # Input params
    u1 = 0.4
    u2 = 0.26
    mstar = 1       # solar masses
    rstar = 1       # solar radii
    rplanet = 0.1   # fraction of stellar radius
    b0 = 0.5        # impact parameter
    P = 50          # orbital period in days
    npts = 25
    time = np.linspace(-0.25, 0.25, npts)

    # Compute the semi-major axis from Kepler's third law in units of rstar
    a = ((P * 86400) ** 2 * (1.32712440018e20 * mstar) /
         (4 * np.pi ** 2)) ** (1. / 3.) / (6.957e8 * rstar)

    # Get the inclination in degrees
    inc = np.arccos(b0 / a) * 180 / np.pi

    # Compute the flux from numerical integration
    f = 2 * np.pi / P * time
    b = a * np.sqrt(1 - np.sin(np.pi / 2. + f) ** 2
                    * np.sin(inc * np.pi / 180) ** 2)
    nF = np.zeros_like(time)
    for i in range(npts):
        nF[i] = NumericalFlux(b[i], rplanet, u1, u2)
    nF /= np.nanmax(nF)
    den = (1 - nF)
    den[den == 0] = 1e-10

    # Compute the starry flux
    # Instantiate a second-order map and a third-order map with u(3) = 0
    # The second-order map is optimized for speed and uses different
    # equations, but they should yield identical results.
    for lmax in [2, 3]:
        star = Primary(lmax)
        star[1] = u1
        star[2] = u2
        planet = Secondary()
        planet.r = rplanet
        planet.a = a
        planet.inc = inc
        planet.porb = P
        planet.lambda0 = 90
        system = System(star, planet)
        system.compute(time)
        sF = np.array(star.lightcurve)
        sF /= sF[0]

        # Compute the error, check that it's better than 1 ppb
        error = np.max((np.abs(nF - sF) / den)[np.where(nF < 1)])
        assert error < 1e-9


if __name__ == "__main__":
    test_transits()
