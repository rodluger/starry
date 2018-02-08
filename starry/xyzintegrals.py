"""Compute the xyz terms of the polynomial integral matrix `S`."""
from .gmatrix import G
from .zintegral import MandelAgolFlux
import numpy as np
from scipy.special import binom
from scipy.special import gamma
from scipy.special import ellipe, ellipk


__all__ = ["Sxyz"]


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


def E1(b, r):
    """Return the first elliptic function."""
    s = (1 - r ** 2 - b ** 2) / (2 * b * r)
    A = (2 * b * r) ** 1.5
    if s < 1:
        return A * (1 - s) / np.sqrt(2) * ellipk(0.5 * (1 + s))
    else:
        return A * (1 - s) / np.sqrt(1 + s) * ellipk(2 / (1 + s))


def E2(b, r):
    """Return the second elliptic function."""
    s = (1 - r ** 2 - b ** 2) / (2 * b * r)
    A = (2 * b * r) ** 1.5
    if s < 1:
        return A * np.sqrt(2) * ellipe(0.5 * (1 + s))
    else:
        return A * np.sqrt(1 + s) * (ellipe(2 / (1 + s)) +
                                     (1 - s) / (1 + s) * ellipk(2 / (1 + s)))


def H(b, r, p, q):
    """Return the `H` function."""
    # Trivial case
    if (p % 2) == 1 or (q % 2) == 1:
        return 0

    # Boundary cases
    u = (1 - r ** 2 - b ** 2) / (2 * b * r)
    if (p == 0) and (q == 0):
        return (2 - 6 * u) / 3. * E1(b, r) + (8 * u) / 3 * E2(b, r)
    elif (p == 0) and (q == 2):
        return (-4 - 12 * u) / 15. * E1(b, r) + \
               (9 + 20 * u + 3 * u ** 2) / 15 * E2(b, r)
    elif (p == 2) and (q == 0):
        return (14 - 18 * u) / 15. * E1(b, r) + \
               (-9 + 20 * u - 3 * u ** 2) / 15 * E2(b, r)
    elif (p == 2) and (q == 2):
        return (5 - 24 * u + 3 * u ** 2) / 105. * E1(b, r) + \
               (29 * u + 3 * u ** 3) / 15 * E2(b, r)

    # General cases
    alpha = q + 2 + (p + q - 2) * (1 - u) / 2
    beta = (3 - q) * (1 - u) / 2
    gamma = 2 * p + q - (p + q - 2) * (1 - u) / 2
    delta = (3 - p) + (p - 3) * (1 - u) / 2
    if (q >= 4):
        return alpha * H(b, r, p, q - 2) + beta * H(b, r, p, q - 4)
    elif (p >= 4):
        return gamma * H(b, r, p - 2, q) + delta * H(b, r, p - 4, q)
    else:
        # DEBUG
        raise Exception("This shouldn't happen!")


def I(b, r, mu, nu):
    """Return the I_{mu,nu} term."""
    res = 0
    for n in range(nu + 1):
        c = binom(nu, n) * (b / r) ** (nu - n)
        for m in range(n + 1):
            res += c * binom(n, m) * (-1) ** (m - n - mu) * \
                   H(b, r, mu + 2 * m, mu + 2 * n - 2 * m)
    return res * 2 ** (mu + 1)


def Sxyz(b, r, i, j):
    """Return an element of `S` corresponding to an `xyz` polynomial."""
    # Check for complete occultation
    if b <= r - 1:
        return 0
    # Check for no occultation
    elif b >= 1 + r:
        # Return the phase curve term
        if (i % 2) == 1 and (j % 2) == 0:
            return 0.5 * np.sqrt(np.pi) * \
                   factorial(0.5 * (i - j - 2)) * \
                   factorial(0.5 * (j - 1)) / \
                   factorial(0.5 * (i + 2))
        else:
            return 0
    # Check for the special case
    elif i == 1:
        return MandelAgolFlux(b, r)

    # General term
    # DEBUG! There's an offset in my notation...
    i -= 1
    if (j < i):
        return -r ** i * I(b, r, i - j, j)
    elif (j % 2) == 0 and (i % 2) == 0:
        return r ** i * I(b, r, i - 2, 2) \
               - b * r ** (i - 1) * I(b, r, i - 2, 1)
    else:
        return r ** i * I(b, r, i - 1, 1) \
               - b * r ** (i - 1) * I(b, r, i - 1, 0)
