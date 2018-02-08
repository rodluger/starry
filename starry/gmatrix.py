"""Compute the Greens change of basis matrix."""
import numpy as np


__all__ = ["G"]


def DF(sz, i, j):
    """Return D ^ F as a polynomial vector."""
    arr = np.zeros(sz)

    if i == j == 0:
        arr[0] = 1

    elif i > j:
        k = j + 0.5 * i * (i + 1)
        arr[int(k)] = j - i - 2

        if (j != i - 1):
            k = j + 0.5 * (i - 1) * (i - 2)
            arr[int(k)] = i - j - 1

            k = j + 2 + 0.5 * i * (i + 1)
            arr[int(k)] = 1 - i + j

    elif i % 2 == 0:

        k = 0.5 * (i - 2) * (i - 1)
        arr[int(k)] = -1

        k = 0.5 * i * (i + 1)
        arr[int(k)] = 1

        k = 2 + 0.5 * i * (i + 1)
        arr[int(k)] = 4

    else:

        k = 1 + 0.5 * i * (i + 1)
        arr[int(k)] = 3

    return arr


def Gz(lmax, tol=1e-10):
    """Return the inverse Greens polynomial matrix for the z terms only."""
    # Matrix order
    N = lmax - 1

    # Number of columns
    ncol = int(0.5 * (N + 1) * (N + 2))

    # The first vector is special
    mat = np.zeros((ncol, ncol))

    # Compute the u and v vectors
    # Note that these are row vectors; we
    # will transpose them below!
    n = 0
    for i in range(N + 1):
        for j in range(i + 1):
            mat[:, n] = DF(ncol, i, j)
            n += 1

    # Compute the inverse
    mat = np.linalg.inv(mat)

    # Remove rounding error; all terms are generally large
    mat[np.abs(mat) < tol] = 0

    return mat


def G(lmax, tol=1e-10):
    """Return the full Greens polynomial matrix."""
    # Compute the z terms matrix
    g = Gz(lmax)

    # Add new identity axes corresponding to the terms
    # that are independent of z
    offset = 0
    inds = np.array([], dtype=int)
    for l in range(1, lmax + 2):
        inds = np.append(inds, np.arange(l) + offset)
        offset = inds[-1]
    g = np.insert(g, inds, 0, axis=0)
    g = np.insert(g, inds, 0, axis=1)
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if (l + m) % 2 == 0:
                g[n, n] = 1
            n += 1

    return g
