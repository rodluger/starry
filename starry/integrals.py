"""Polynomial integral matrix."""
import numpy as np
from scipy.special import binom


__all__ = ["S"]


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


def HEven(phi, p, q):
    """Return the `H` function for 'even' polynomial terms."""
    if (p % 2) != 0:
        return 0
    elif (p == 0) and (q == 0):
        return 2 * phi + np.pi
    elif (p == 0) and (q == 1):
        return -2 * np.cos(phi)
    elif (p >= 2):
        return (2. / (p + q)) * np.cos(phi) ** (p - 1) * \
                                np.sin(phi) ** (q + 1) + \
               (p - 1.) / (p + q) * HEven(phi, p - 2, q)
    else:
        return -(2. / (p + q)) * np.cos(phi) ** (p + 1) * \
                                 np.sin(phi) ** (q - 1) + \
                (q - 1.) / (p + q) * HEven(phi, p, q - 2)


def SEven(b, r, i, j):
    """Return the element of `s` for an 'even' polynomial term."""
    # Check for complete occultation
    if b <= r - 1:
        return 0
    # Check for no occultation
    elif b >= 1 + r:
        # TODO!
        return np.nan
    lam = Lam(b, r)
    phi = Phi(b, r)
    Hlam = HEven(lam, i - j + 2, j)
    Hphi = r ** (i + 2) * np.sum([
        binom(j, n) * (b / r) ** (j - n) * HEven(phi, i - j + 2, n)
        for n in range(j + 1)])
    return 1. / (i - j + 1.) * (Hphi - Hlam)


def SOdd(b, r, i, j):
    """Return the element of `s` for an 'odd' polynomial term."""
    # TODO!
    return 0


def S(lmax, b, r):
    """Return the polynomial integral vector."""
    vec = np.zeros((lmax + 1) ** 2, dtype=float)
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            # Compute the power indices
            k = int((1 - (-1) ** (m + l)) / 2)
            j = int((m + l - k) / 2)
            i = int(l)
            # Even terms (no z dependence)
            if k == 0:
                vec[n] = SEven(b, r, i, j)
            # Odd terms (linear in z)
            else:
                vec[n] = SOdd(b, r, i, j)
            n += 1
    return vec


print(S(2, 0.4, 0.5))
