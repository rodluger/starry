"""Compute the z term of the polynomial integral matrix `S`."""
import numpy as np
from scipy.special import ellipe, ellipk
from sympy.mpmath import ellippi


__all__ = ["MandelAgolFlux"]


def step(n):
    """Return the Heaviside step function Theta."""
    if n < 0:
        return 0
    else:
        return 1


def MandelAgolFlux(b, r):
    """Return the Mandel & Agol (2002) occultation flux."""
    # HACK: Nudge b?
    tol = 1e-5
    if b == 0:
        b = tol

    # Elliptic argument
    u = (1 - r ** 2 - b ** 2) / (2 * b * r)
    ksq = (1 + u) / 2.

    # HACK: Nudge ksq?
    if ksq == 1:
        ksq = 1 - tol

    # Two cases
    xi = 2 * b * r * (4 - 7 * r ** 2 - b ** 2)
    cpi = 3 * (b + r) / (b - r)
    if ksq < 1:
        ck = -3 + 12 * r ** 2 - 10 * b ** 2 * r ** 2 - 6 * r ** 4 + xi
        ce = -2 * xi
        K = ellipk(ksq)
        E = ellipe(ksq)
        PI = ellippi(1 - (b - r) ** -2, ksq)
        lam = (ck * K + ce * E + cpi * PI) / (9 * np.pi * np.sqrt(b * r))
    else:
        ck = 1 - 5 * b ** 2 + r ** 2 + (r ** 2 - b ** 2) ** 2
        ce = -2 * xi * ksq
        n = 1. / (ksq - 1. / (4 * b * r))
        K = ellipk(1. / ksq)
        E = ellipe(1. / ksq)
        PI = ellippi(n, 1. / ksq)
        lam = 2 * (ck * K + ce * E + cpi * PI) / \
                  (9 * np.pi * np.sqrt((1 - b + r) * (1 + b - r)))

    # Total flux from S10 (Equation 28)
    tot = 2 * np.pi / 3

    # Normalized flux during occultation
    F = 1 - 1.5 * lam - step(r - b)
    return tot * F
