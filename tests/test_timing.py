"""Test the code run time."""
from starry2 import Map
import numpy as np
import time


def test_small(benchmark=0.1):
    """Small occultor."""
    # Let's do the l = 5 Earth
    map = Map(5)
    map.load_image('earth')

    # Occultation properties
    npts = 10550
    ro = 0.1
    xo = np.linspace(-1 - ro, 1 + ro, npts)
    yo = np.linspace(-0.1, 0.1, npts)
    theta = np.linspace(0, 90, npts)
    map.axis = [1, 1, 1] / np.sqrt(3)

    # Analytical and numerical fluxes
    t = np.zeros(10)
    for i in range(10):
        tstart = time.time()
        map.flux(theta=theta, xo=xo, yo=yo, ro=ro)
        t[i] = time.time() - tstart
    t = np.mean(t)

    # Print
    print("Time [Benchmark]: %.3f [%.3f]" % (t, benchmark))


def test_large(benchmark=0.1):
    """Large occultor."""
    # Let's do the l = 5 Earth
    map = Map(5)
    map.load_image('earth')

    # Occultation properties
    npts = 10350
    ro = 10
    xo = np.linspace(ro - 1, ro + 1, npts)
    yo = np.linspace(-0.1, 0.1, npts)
    theta = np.linspace(0, 90, npts)
    map.axis = [1, 1, 1] / np.sqrt(3)

    # Analytical and numerical fluxes
    t = np.zeros(10)
    for i in range(10):
        tstart = time.time()
        map.flux(theta=theta, xo=xo, yo=yo, ro=ro)
        t[i] = time.time() - tstart
    t = np.mean(t)

    # Print
    print("Time [Benchmark]: %.3f [%.3f]" % (t, benchmark))


if __name__ == "__main__":
    test_small()
    test_large()
