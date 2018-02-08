"""Change of basis from spherical harmonics to polynomials."""
from .gmatrix import G
import numpy as np
from scipy.special import gamma


__all__ = ["A"]


def factorial(n):
    """Define the factorial for fractions and negative numbers."""
    return gamma(n + 1)


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


def A(lmax):
    """Return the complete basis change matrix."""
    # Spherical harmonics to polynomials
    A1 = np.zeros(((lmax + 1) ** 2, (lmax + 1) ** 2), dtype=float)
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            A1[:(l + 1) ** 2, n] = Y(l, m)
            n += 1

    # Polynomials to the Greens basis
    A2 = G(lmax)

    return np.dot(A2, A1)
