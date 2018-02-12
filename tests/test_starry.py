"""Test the main STARRY code."""
import sys; sys.path.insert(1, "../")
from starry import starry, ylm
import numpy as np


def test_occultation():
    """Test an occultation calculation."""
    # Let's do Y_{3,2}
    y = ylm(3, 2)
    s = starry(y)

    # Occultation params
    npts = 25
    x0 = np.linspace(-1.5, 1.5, npts)
    y0 = 0.5
    r = 0.3
    theta = np.linspace(0, np.pi, npts)
    u = [0, 1, 0]

    # Compute the analytic flux
    flux = s.flux(u, theta, x0, y0, r)

    # Compute the numerical flux
    flux_n = s.flux(u, theta, x0, y0, r, debug=True)

    # Compute the error
    diff = (flux - flux_n)

    assert np.max(np.abs(diff)) < 0.01, "Flux mismatch."
