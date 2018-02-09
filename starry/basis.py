"""Change of basis from spherical harmonics to polynomials."""
from .utils import factorial
import numpy as np


__all__ = ["A"]


def polybasis(x, y, lmax):
    """Return a vector corresponding to the polynomial basis."""
    z = np.sqrt(1 - x ** 2 - y ** 2)
    res = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m
            if (nu % 2) == 0:
                res.append(x ** (mu / 2) * y ** (nu / 2))
            else:
                res.append(x ** ((mu - 1) / 2) * y ** ((nu - 1) / 2) * z)
    return np.array(res)


def evaluate_poly(p, x, y):
    """Evaluate a polynomial vector at a given (x, y) coordinate."""
    lmax = int(np.sqrt(len(p)) - 1)
    return np.dot(p, polybasis(x, y, lmax))


def C(p, q, k):
    """Contraction coefficient."""
    return factorial(0.5 * k) / (factorial(0.5 * q) *
                                 factorial(0.5 * (k - p)) *
                                 factorial(0.5 * (p - q)))


def Norm(l, m):
    """Return the normalization constant for a Ylm."""
    return np.sqrt(1. / (4 * np.pi)) * \
        np.sqrt((2 - (m == 0)) *
                (2 * l + 1) *
                factorial(l - abs(m)) /
                factorial(l + abs(m)))


def B(l, m, j, k):
    """Return the coefficient for a Ylm."""
    num = 2 ** l * factorial(m) * factorial(0.5 * (l + m + k - 1))
    den = factorial(j) * factorial(k) * factorial(m - j) * \
        factorial(l - m - k) * factorial(0.5 * (-l + m + k - 1))
    return num / den


def Lijk(l, m, i, j, k):
    """Return the ijk tensor element of the spherical harmonic Ylm."""
    # Kronecker delta
    if (i == np.abs(m) + k) and (j <= np.abs(m)):
        if (m >= 0) and (j % 2 == 0):
            return (-1) ** (j / 2) * Norm(l, m) * B(l, m, j, k)
        elif (m < 0) and (j % 2 == 1):
            return (-1) ** ((j - 1) / 2) * Norm(l, -m) * B(l, -m, j, k)
        else:
            return 0
    else:
        return 0


def Y(l, m):
    """Return a vector of polynomial coefficients corresponding to a Ylm."""
    # Initialize the contracted polynomial tensor
    Ylm = np.zeros((l + 1, l + 1, 2))
    for k in range(0, l + 1):
        for i in range(k, l + 1):
            for j in range(0, i - k + 1):
                coeff = Lijk(l, m, i, j, k)
                if coeff:
                    # 1 or z
                    if (k == 0) or (k == 1):
                        Ylm[i, j, k] += coeff
                    # Even power of z
                    elif (k % 2) == 0:
                        for p in range(0, k + 1, 2):
                            for q in range(0, p + 1, 2):
                                ip = i - k + p
                                jp = j + q
                                Ylm[ip, jp, 0] += (-1) ** (p / 2) * \
                                    C(p, q, k) * coeff
                    # Odd power of z
                    else:
                        for p in range(0, k, 2):
                            for q in range(0, p + 1, 2):
                                ip = i - k + p + 1
                                jp = j + q
                                Ylm[ip, jp, 1] += (-1) ** (p / 2) * \
                                    C(p, q, k - 1) * coeff

    # Now we contract the tensor down to a vector
    vec = np.zeros((l + 1) ** 2, dtype=float)
    n = 0
    for i in range(l + 1):
        for j in range(i + 1):
            vec[n] = Ylm[i, j, 0]
            n += 1
            if (j < i):
                vec[n] = Ylm[i, j, 1]
                n += 1

    return vec


def DG(sz, i, j):
    """Return D ^ G as a polynomial vector."""
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
            mat[:, n] = DG(ncol, i, j)
            n += 1

    # Compute the inverse
    mat = np.linalg.inv(mat)

    # Remove rounding error; all terms are generally large
    mat[np.abs(mat) < tol] = 0

    return mat


def Y2P(lmax):
    """
    Return the polynomial matrix.

    This matrix performs a change of basis from spherical harmonics (Y) to
    polynomials (P) of the form x^i y^j z^k.
    """
    # Spherical harmonics to polynomials
    p = np.zeros(((lmax + 1) ** 2, (lmax + 1) ** 2), dtype=float)
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            p[:(l + 1) ** 2, n] = Y(l, m)
            n += 1
    return p


def y2p(y):
    """Rotate the Ylm vector `y` into the polynomial basis."""
    lmax = int(np.sqrt(len(y)) - 1)
    return np.dot(Y2P(lmax), y)


def P2G(lmax):
    """
    Return the Greens matrix.

    This matrix performs a change of basis from polynomials (P) to
    normalized Greens polynomials (G) used to solve the primitive integrals.
    """
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
                # The coefficient of this term is (mu + 2) / 2
                # So its inverse is
                g[n, n] = 2 / (l - m + 2)
            n += 1
    return g


def p2g(p):
    """Rotate the polynomial vector `p` into the Greens basis."""
    lmax = int(np.sqrt(len(p)) - 1)
    return np.dot(P2G(lmax), p)


def A(lmax):
    """
    Return the complete basis change matrix `A`.

    This matrix performs a change of basis from spherical harmonics
    to normalized Greens polynomials.
    """
    return np.dot(P2G(lmax), Y2P(lmax))
