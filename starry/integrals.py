"""Compute the polynomial integral matrix `S`."""
from .xyintegrals import Sxy
from .xyzintegrals import Sxyz
import numpy as np


__all__ = ["S"]


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
            # Terms with no z dependence
            if k == 0:
                vec[n] = Sxy(b, r, i, j)
            # Terms that are linear in z
            else:
                vec[n] = Sxyz(b, r, i, j)
            n += 1
    return vec
