"""User-facing routines for STARRY."""
import numpy as np
from .rotation import R, Rz
from .basis import A, evaluate_poly, y2p
from .integrals import S, brute

__all__ = ["starry", "ylm"]


def ylm(l, m):
    """Return a Ylm vector corresponding to the spherical harmonic Y_{l,m}."""
    assert (l >= 0), "Order `l` must be greater than or equal to zero."
    assert (np.abs(m) <= l), "Degree `m` must be in the range `[-l, l]`."
    y = np.zeros((l + 1) ** 2, dtype=float)
    y[l ** 2 + l + m] = 1
    return y


class starry(object):
    """The main STARRY interface."""

    def __init__(self, y, tol=1e-15):
        """Initialize and precompute some stuff."""
        # These are the spherical harmonic coefficients
        self.y = np.array(y, dtype=float)

        # Integration tolerance
        self.tol = 1e-15

        # Check that the length of y is a perfect square
        self.lmax = np.sqrt(len(y)) - 1
        assert self.lmax % 1 == 0, "Invalid dimensions for `y`."
        self.lmax = int(self.lmax)

        # Pre-compute the basis change matrix
        self.A = A(self.lmax)

    def flux(self, u, theta, x0=None, y0=None, r=None, debug=False, res=100):
        """Return the flux visible from the map."""
        # Is this an occultation?
        if (x0 is None) or (y0 is None) or (r is None):
            # Vectorize the phase
            occultation = False
            theta = np.atleast_1d(theta)
            npts = len(theta)
            static_map = False
        else:
            # Vectorize the inputs
            occultation = True
            x0 = np.atleast_1d(x0)
            y0 = np.atleast_1d(y0)
            r = np.atleast_1d(r)
            theta = np.atleast_1d(theta)
            npts = max(len(x0), len(y0), len(r), len(theta))
            if len(x0) == 1:
                x0 = np.ones(npts, dtype=float) * x0[0]
            if len(y0) == 1:
                y0 = np.ones(npts, dtype=float) * y0[0]
            if len(r) == 1:
                r = np.ones(npts, dtype=float) * r[0]
            if len(theta) == 1:
                static_map = True
                theta = np.ones(npts, dtype=float) * theta[0]
            else:
                static_map = False
            assert len(x0) == len(y0) == len(r) == len(theta), \
                "Invalid dimensions for `x0`, `y0`, `r`, and/or `theta`."

        # Is the map static during the occultation?
        # If so, pre-compute the phase rotation matrix
        if static_map:
            R1 = R(self.lmax, u, np.cos(theta[0]),
                   np.sin(theta[0]), tol=self.tol)
            Ry = np.dot(R1, self.y)

        # Iterate through the timeseries
        F = np.zeros(npts, dtype=float)
        for n in range(npts):

            # Compute the impact parameter
            if occultation:
                b = np.sqrt(x0[n] ** 2 + y0[n] ** 2)
            else:
                b = np.inf

            # Rotate the map so the occultor is along the y axis
            if (b > self.tol) and (b < 1 + r[n]) and (not debug):
                sinvt = x0[n] / b
                cosvt = y0[n] / b
                R2 = Rz(self.lmax, cosvt, sinvt, tol=self.tol)
            # If there's no occultation, if b is zero, or if we
            # are computing the flux numerically,
            # we don't need to rotate
            else:
                R2 = np.eye((self.lmax + 1) ** 2)

            # If the map is static, we need only rotate about z
            if static_map:
                RRy = np.dot(R2, Ry)
            # For a rotating map, we need to apply two rotations
            else:
                R1 = R(self.lmax, u, np.cos(theta[n]),
                       np.sin(theta[n]), tol=self.tol)
                Ry = np.dot(R1, self.y)
                RRy = np.dot(R2, Ry)

            # Are we computing the flux numerically...
            if debug:
                F[n] = brute(RRy, x0[n], y0[n], r[n], res=res)
            # ...or analytically?
            else:
                # Convert to the Greens basis
                ARRy = np.dot(self.A, RRy)
                # Compute the integrals and solve for the flux array
                sT = S(self.lmax, b, r[n])
                F[n] = np.dot(sT, ARRy)

        return F

    def render(self, u, theta, x0=None, y0=None, r=None, res=100):
        """Return a pixelized image of the visible portion of the map."""
        # Is this an occultation?
        if (x0 is None) or (y0 is None) or (r is None):
            x0 = 0
            y0 = 0
            r = 0

        # Rotate the map
        y = np.dot(R(self.lmax, u, np.cos(theta), np.sin(theta), tol=self.tol),
                   self.y)

        # Convert it to a polynomial
        poly = y2p(y)

        # Render it
        F = np.zeros((res, res)) * np.nan
        for i, x in enumerate(np.linspace(-1, 1, res)):
            for j, y in enumerate(np.linspace(-1, 1, res)):
                # Are we inside the body?
                if (x ** 2 + y ** 2 < 1):
                    # Are we outside the occultor?
                    if (x - x0) ** 2 + (y - y0) ** 2 > r ** 2:
                        F[j][i] = evaluate_poly(poly, x, y)

        return F
