"""General utility functions."""
import numpy as np
from scipy.special import gamma
from scipy.special import ellipe, ellipk


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
