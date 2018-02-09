"""General utility functions."""
import numpy as np
from scipy.special import gamma
from scipy.special import ellipe, ellipk
import re


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


def greens_basis(lmax):
    """Return the greens polynomial basis as a LaTeX table."""
    table = [["" for m in range(-l, l + 1)] for l in range(lmax)]
    for l in range(lmax):
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m
            if (nu % 2) == 0:
                g = r"\frac{1}{%d}x^{%d}y^{%d}" % \
                    (mu / 2 + 1, mu / 2, nu / 2)
            elif nu == 1 and mu == 1:
                g = "z"
            elif mu > 1:
                if mu != 3:
                    g = "z(%dx^{%d}y^{%d} - %dx^{%d}y^{%d} - %dx^{%d}y^{%d})" \
                        % ((mu - 3) / 2, (mu - 5) / 2, (nu - 1) / 2,
                           (mu - 3) / 2, (mu - 5) / 2, (nu + 3) / 2,
                           (mu + 3) / 2, (mu - 1) / 2, (nu - 1) / 2)
                else:
                    g = "z(-%dx^{%d}y^{%d})" \
                        % ((mu + 3) / 2, (mu - 1) / 2, (nu - 1) / 2)
            elif mu == 1 and (l % 2) == 1:
                g = "z(-x^{%d} + x^{%d} + 4x^{%d}y^{2})" % \
                    ((l - 3), (l - 1), (l - 3))
            else:
                g = "z(3x^{%d}y)" % (l - 2)

            # Clean things up a bit
            g = g.replace(r"\frac{1}{1}", "")
            g = g.replace("1x^{0}y^{0}", "1")
            g = g.replace("x^{0}y^{0}", "1")
            g = g.replace("x^{0}y", "y")
            g = g.replace("x^{0} ", "1 ")
            g = g.replace("y^{0}", "")
            g = g.replace("x^{1}", "x")
            g = g.replace("y^{1}", "y")
            g = g.replace("(1y", "(y")
            g = g.replace(" 1y", " y")
            # Add to the table
            table[l][nu] = g

    return table
