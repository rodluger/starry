"""Test the main STARRY code."""
from starry import starry
from starry.utils import prange
import numpy as np
np.random.seed(1234)


def diff(l, m, x0, y0, r, res=300):
    """Test an occultation of a Ylm."""
    s = starry(l)
    s[l, m] = 1

    # Compute the analytic flux
    flux = s.flux(x0=x0, y0=y0, r=r)

    # Compute the numerical flux
    flux_n = s.flux(x0=x0, y0=y0, r=r, res=res, debug=True)

    # Compute the error
    diff = (flux - flux_n)

    return np.abs(diff)


def test_occult_ylm(lmax=5, res=300, tol=0.01):
    """Test several ylm occultations."""
    # Loop over all Ylms
    for n in prange(lmax ** 2 + 2 * lmax + 1):

        # Get l and m
        l = int(np.floor(np.sqrt(n)))
        m = n - l ** 2 - l

        # Randomize r in the range (0, 2)
        r = 2 * np.random.random()

        # Randomize x0 and y0
        x0 = 99
        y0 = 99
        while x0 ** 2 + y0 ** 2 > (1 + r) ** 2:
            x0 = np.random.random()
            y0 = np.random.random()

        d = diff(l, m, x0, y0, r, res=res)
        if d > tol:
            print("ERROR: Flux mismatch.")
            print("l:    %d" % l)
            print("m:    %d" % m)
            print("x0:   %.5f" % x0)
            print("y0:   %.5f" % y0)
            print("r:    %.5f" % r)
            print("res:  %.5f" % r)
            print("diff: %.5f" % d)
            raise AssertionError("Flux mismatch.")
