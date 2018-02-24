"""Compute the polynomial integral matrix `S`."""
from .utils import Phi, Lam, E1, E2, factorial, prange
from .zintegral import MandelAgolFlux
from .basis import evaluate_poly, y2p
import numpy as np
from scipy.special import binom
from scipy.integrate import dblquad


__all__ = ["S", "brute"]


def polynomial_flux(i, j, k):
    """Return the total flux from the polynomial term x^i y^j z^k."""
    if (i % 2) == 1 or (j % 2) == 1:
        return 0
    elif k == 0:
        return factorial(0.5 * (i - 1)) * factorial(0.5 * (j - 1)) / \
               factorial(0.5 * (i + j + 2))
    elif k == 1:
        return 0.5 * np.sqrt(np.pi) * \
                     factorial(0.5 * (i - 1)) * factorial(0.5 * (j - 1)) / \
                     factorial(0.5 * (i + j + 3))
    else:
        raise ValueError("The z power `k` must be 0 or 1.")


def greens_flux(l, m):
    """Return the total flux from the (l, m) term in the Greens basis."""
    mu = l - m
    nu = l + m

    # Case A
    if (nu % 2) == 0:
        return 0.5 * (mu + 2) * polynomial_flux(mu / 2, nu / 2, 0)
    # Case B
    elif (nu == 1) and (mu == 1):
        return polynomial_flux(0, 0, 1)
    # Case C
    elif (mu > 1):
        res = 0
        res += (mu - 3) / 2 * polynomial_flux((mu - 5) / 2, (nu - 1) / 2, 1)
        res -= (mu - 3) / 2 * polynomial_flux((mu - 5) / 2, (nu + 3) / 2, 1)
        res -= (mu + 3) / 2 * polynomial_flux((mu - 1) / 2, (nu - 1) / 2, 1)
        return res
    # Case D
    elif (l % 2) == 1:
        res = 0
        res -= polynomial_flux(l - 3, 0, 1)
        res += polynomial_flux(l - 1, 0, 1)
        res += 4 * polynomial_flux(l - 3, 2, 1)
        return res
    # Case E
    elif (l % 2) == 0:
        return 3 * polynomial_flux(l - 2, 1, 1)
    # Should never reach here
    else:
        raise Exception("Undefined case!")


def I(u, v, nu, phi, b, r):
    """Return the function `I_{u, v}^{nu}`."""
    if b == 0:
        return H(u, v, nu, phi, 0, r)
    else:
        return np.sum([
            binom(v, n) * (b / r) ** (v - n) * H(u, n, nu, phi, b, r)
            for n in range(v + 1)])


def H(u, n, nu, phi, b, r):
    """Return the function `H_{u, n}^{nu}`."""
    # No z dependence
    if (nu % 2) == 0:
        if (u % 2) != 0:
            return 0
        elif (u == 0) and (n == 0):
            return 2 * phi + np.pi
        elif (u == 0) and (n == 1):
            return -2 * np.cos(phi)
        elif (u >= 2):
            return (2. / (u + n)) * np.cos(phi) ** (u - 1) * \
                                    np.sin(phi) ** (n + 1) + \
                   (u - 1.) / (u + n) * H(u - 2, n, nu, phi, b, r)
        else:
            return -(2. / (u + n)) * np.cos(phi) ** (u + 1) * \
                                     np.sin(phi) ** (n - 1) + \
                    (n - 1.) / (u + n) * H(u, n - 2, nu, phi, b, r)
    # Linear in z
    else:
        # Avoid the singularity
        if b == 0:
            return (1 - r ** 2) ** 1.5 * H(u, n, nu + 1, phi, b, r)
        else:
            return 2 ** (u + 1) * np.sum([
                binom(n, m) * (-1) ** (m - n - u) *
                J(u + 2 * m, u + 2 * n - 2 * m, b, r)
                for m in range(n + 1)])


def J(p, q, b, r):
    """Return the function `J_{p, q}`."""
    # Trivial case
    if (q % 2) == 1:
        return 0

    # Forbidden case
    if (p % 2) == 1:
        raise Exception("Undefined case!")

    # Boundary cases
    u = (1 - r ** 2 - b ** 2) / (2 * b * r)
    if (p == 0) and (q == 0):
        return (2 - 6 * u) / 3. * E1(b, r) + (8 * u) / 3. * E2(b, r)
    elif (p == 0) and (q == 2):
        return (-4 - 12 * u) / 15. * E1(b, r) + \
               (9 + 20 * u + 3 * u ** 2) / 15. * E2(b, r)
    elif (p == 2) and (q == 0):
        return (14 - 18 * u) / 15. * E1(b, r) + \
               (-9 + 20 * u - 3 * u ** 2) / 15. * E2(b, r)
    elif (p == 2) and (q == 2):
        return (5 - 24 * u + 3 * u ** 2) / 105. * E1(b, r) + \
               (29 * u + 3 * u ** 3) / 105. * E2(b, r)

    # General cases
    alpha = q + 2 + (p + q - 2) * (1 - u) / 2
    beta = (3 - q) * (1 - u) / 2
    gamma = 2 * p + q - (p + q - 2) * (1 - u) / 2
    delta = (3 - p) + (p - 3) * (1 - u) / 2
    if (q >= 4):
        return (alpha * J(p, q - 2, b, r) + beta * J(p, q - 4, b, r)) / \
               (p + q + 3)
    elif (p >= 4):
        return (gamma * J(p - 2, q, b, r) + delta * J(p - 4, q, b, r)) / \
               (p + q + 3)
    # Should never reach here
    else:
        raise Exception("Undefined case!")


def P(l, m, b, r):
    """Return the primitive integral `P`."""
    mu = l - m
    nu = l + m
    phi = Phi(b, r)

    # Case A
    if (nu % 2) == 0:
        return r ** (l + 2) * \
               I(int((mu + 4) / 2), int(nu / 2), nu, phi, b, r)
    # Case B
    elif (nu == 1) and (mu == 1):
        raise Exception("Undefined case!")
    # Case C
    elif (mu > 1):
        return r ** (l - 1) * \
               I(int((mu - 1) / 2), int((nu - 1) / 2), nu, phi, b, r)
    # Case D
    elif (l % 2) == 1:
        return b * r ** (l - 2) * I(l - 3, 1, nu, phi, b, r) - \
                   r ** (l - 1) * I(l - 3, 2, nu, phi, b, r)
    # Case E
    elif (l % 2) == 0:
        return b * r ** (l - 2) * I(l - 2, 0, nu, phi, b, r) - \
                   r ** (l - 1) * I(l - 2, 1, nu, phi, b, r)
    # Should never reach here
    else:
        raise Exception("Undefined case!")


def Q(l, m, b, r):
    """Return the primitive integral `Q`."""
    mu = l - m
    nu = l + m
    lam = Lam(b, r)

    # Case A
    if (nu % 2) == 0:
        return I(int((mu + 4) / 2), int(nu / 2), nu, lam, 0, 1)
    # Case B
    elif (nu == 1) and (mu == 1):
        raise Exception("Undefined case!")
    # All other cases
    else:
        return 0


def slm(l, m, b, r):
    """Return the (l, m) element of the vector `s`."""
    # Is this a complete occultation?
    if b <= r - 1:
        return 0

    # Is there no occultation happening?
    # If that's the case, the solutions are easy
    elif b >= 1 + r:
        return greens_flux(l, m)

    # TODO: Pre-compute E1 and E2 here

    # Is this the lone `z` term?
    elif (l == 1) and (m == 0):
        return MandelAgolFlux(b, r)

    # We need to compute the flux using Green's theorem
    return -(P(l, m, b, r) - Q(l, m, b, r))


def S(lmax, b, r):
    """Return the integral solution vector `s`."""
    vec = np.zeros((lmax + 1) ** 2, dtype=float)
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            vec[n] = slm(l, m, b, r)
            n += 1
    return vec


def brute(ylm, x0, y0, r, res=100):
    """Compute the occultation flux for a Ylm vector the brute-force way."""
    # Convert the Ylm vector to a polynomial vector
    p = y2p(ylm)

    # Define our map
    def func(y, x):
        return evaluate_poly(p, x, y)

    # Lower integration limit for total flux
    def y1(x):
        return -np.sqrt(1 - x ** 2)

    # Upper integration limit for total flux
    def y2(x):
        return np.sqrt(1 - x ** 2)

    # Compute the total flux
    flux, _ = dblquad(func, -1, 1, y1, y2)

    # Occultor is smaller than occulted
    if r < 1:

        # Subtract off the occulted flux
        dA = (2 * r / res) ** 2
        x = np.linspace(x0 - r, x0 + r, res)
        y = np.linspace(y0 - r, y0 + r, res)
        for i in range(res):
            for j in range(res):
                # If inside the occultor
                if (x[i] - x0) ** 2 + (y[j] - y0) ** 2 < r ** 2:
                    # If inside the occulted
                    if (x[i] ** 2 + y[j] ** 2 < 1):
                        flux -= func(y[j], x[i]) * dA

    # Occultor is larger than occulted
    else:

        # Subtract off the occulted flux
        dA = (2 / res) ** 2
        x = np.linspace(-1, 1, res)
        y = np.linspace(-1, 1, res)
        for i in range(res):
            for j in range(res):
                # If inside the occulted
                if (x[i] ** 2 + y[j] ** 2 < 1):
                    # If inside the occultor
                    if (x[i] - x0) ** 2 + (y[j] - y0) ** 2 < r ** 2:
                        flux -= func(y[j], x[i]) * dA

    return flux
