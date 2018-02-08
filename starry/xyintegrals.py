"""Compute the xy terms of the polynomial integral matrix `S`."""
import numpy as np
from scipy.special import binom
from scipy.special import gamma


__all__ = ["Sxy"]


def factorial(n):
    """Define the factorial for fractions and negative numbers."""
    return gamma(n + 1)


def Lam(b, r):
    """Return the angle lambda."""
    if np.abs(1 - r) < b and b < 1 + r:
        v = b + (1 - r ** 2 - b ** 2) / (2 * b)
        return np.arcsin(v)
    else:
        return np.pi / 2


def Phi(b, r):
    """Return the angle phi."""
    if np.abs(1 - r) < b and b < 1 + r:
        u = (1 - r ** 2 - b ** 2) / (2 * b * r)
        return np.arcsin(u)
    else:
        return np.pi / 2


def H(phi, p, q):
    """Return the `H` function."""
    if (p % 2) != 0:
        return 0
    elif (p == 0) and (q == 0):
        return 2 * phi + np.pi
    elif (p == 0) and (q == 1):
        return -2 * np.cos(phi)
    elif (p >= 2):
        return (2. / (p + q)) * np.cos(phi) ** (p - 1) * \
                                np.sin(phi) ** (q + 1) + \
               (p - 1.) / (p + q) * H(phi, p - 2, q)
    else:
        return -(2. / (p + q)) * np.cos(phi) ** (p + 1) * \
                                 np.sin(phi) ** (q - 1) + \
                (q - 1.) / (p + q) * H(phi, p, q - 2)


def Sxy(b, r, i, j):
    """Return an element of `S` corresponding to an `xy` polynomial."""
    # Check for complete occultation
    if b <= r - 1:
        return 0
    # Check for no occultation
    elif b >= 1 + r:
        # Return the phase curve term
        if (i % 2) == 0 and (j % 2) == 0:
            return factorial(0.5 * (i - j - 1)) * factorial(0.5 * (j - 1)) / \
                   factorial(0.5 * (i + 2))
        else:
            return 0

    # Let's compute the flux using Green's theorem
    lam = Lam(b, r)
    phi = Phi(b, r)
    Hlam = H(lam, i - j + 2, j)
    Hphi = r ** (i + 2) * np.sum([
        binom(j, n) * (b / r) ** (j - n) * H(phi, i - j + 2, n)
        for n in range(j + 1)])
    return -1. / (i - j + 1.) * (Hphi - Hlam)
