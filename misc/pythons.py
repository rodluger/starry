"""Python version of the s vector for debugging."""
import numpy as np
from scipy.misc import comb
from scipy.special import ellipk, ellipe
import matplotlib.pyplot as pl
import starry


def choose(n, k):
    """Return n choose k."""
    return comb(n, k, exact=True)


def is_even(n):
    """Return true if n is even."""
    if ((n % 2) != 0):
        return False
    else:
        return True


def ComputeSStarry(barr, r, l, m):
    """Compute s with starry."""
    map = starry.Map(l)
    map.optimize = False
    for ll in range(l + 1):
        for mm in range(-ll, ll + 1):
            map[ll, mm] = 1
    s = np.zeros_like(barr)
    n = l ** 2 + l + m
    for i in range(len(barr)):
        map.flux(xo=0, yo=barr[i], ro=r)
        s[i] = map.s[n]
    return s


def ComputeSExact(barr, r, l, m):
    """Compute s with starry multiprecision."""
    map = starry.Map(l)
    for ll in range(l + 1):
        for mm in range(-ll, ll + 1):
            map[ll, mm] = 1
    s = np.zeros_like(barr)
    n = l ** 2 + l + m
    for i in range(len(barr)):
        map.flux_mp(xo=0, yo=barr[i], ro=r)
        s[i] = map.s_mp[n]
    return s


def ComputeS(barr, r, l, m):
    """Compute s."""
    s = np.zeros_like(barr)
    for ind, b in enumerate(barr):
        ksq = (1 - (b - r)) * (1 + (b - r)) / (4 * b * r)
        if ((abs(1 - r) < b) and (b < 1 + r)):
            sinphi = 2 * (ksq - 0.5)
            sinlam = 0.5 * ((1. / b) + (b - r) * (1. + r / b))
            phi = np.arcsin(sinphi)
            lam = np.arcsin(sinlam)
        else:
            phi = np.pi / 2
            lam = np.pi / 2

        def EllipticK():
            """Return EllipticK."""
            if (b == 0) or (ksq == 1):
                return 0
            elif ksq < 1:
                return ellipk(ksq)
            else:
                return ellipk(1. / ksq)

        def EllipticE():
            """Return EllipticE."""
            if (b == 0):
                return 0
            elif (ksq == 1):
                return 1
            elif (ksq < 1):
                return ellipe(ksq)
            else:
                return ellipe(1. / ksq)

        def E1():
            """Return E1."""
            if (b == 0) or (ksq == 1):
                return 0
            elif (ksq < 1):
                return (1 - ksq) * EllipticK()
            else:
                return (1 - ksq) / np.sqrt(ksq) * EllipticK()

        def E2():
            """Return E2."""
            if (b == 0):
                return 0
            elif (ksq == 1):
                return 1
            elif (ksq < 1):
                return EllipticE()
            else:
                return np.sqrt(ksq) * EllipticE() + \
                       (1 - ksq) / np.sqrt(ksq) * EllipticK()

        def H(u, v):
            """Return H."""
            if (not is_even(u)):
                return 0
            elif ((u == 0) and (v == 0)):
                return 2 * lam + np.pi
            elif ((u == 0) and (v == 1)):
                return -2 * np.cos(lam)
            elif (u >= 2):
                return (2 * np.cos(lam) ** (u - 1) * np.sin(lam) ** (v + 1) +
                        (u - 1) * H(u - 2, v)) / (u + v)
            else:
                return (-2 * np.cos(lam) ** (u + 1) * np.sin(lam) ** (v - 1) +
                        (v - 1) * H(u, v - 2)) / (u + v)

        def I(u, v):
            """Return I."""
            if (not is_even(u)):
                return 0
            elif ((u == 0) and (v == 0)):
                return 2 * phi + np.pi
            elif ((u == 0) and (v == 1)):
                return -2 * np.cos(phi)
            elif (u >= 2):
                return (2 * np.cos(phi) ** (u - 1) * np.sin(phi) ** (v + 1) +
                        (u - 1) * I(u - 2, v)) / (u + v)
            else:
                return (-2 * np.cos(phi) ** (u + 1) * np.sin(phi) ** (v - 1) +
                        (v - 1) * I(u, v - 2)) / (u + v)

        def M(p, q):
            """Return M."""
            if (not is_even(p) or not is_even(q)):
                return 0
            elif ((p == 0) and (q == 0)):
                return ((8 - 12 * ksq) * E1() +
                        (-8 + 16 * ksq) * E2()) / 3.
            elif ((p == 0) and (q == 2)):
                return ((8 - 24 * ksq) * E1() +
                        (-8 + 28 * ksq + 12 * ksq ** 2) * E2()) / 15.
            elif ((p == 2) and (q == 0)):
                return ((32 - 36 * ksq) * E1() +
                        (-32 + 52 * ksq - 12 * ksq ** 2) * E2()) / 15.
            elif ((p == 2) and (q == 2)):
                return ((32 - 60 * ksq + 12 * ksq ** 2) * E1() +
                        (-32 + 76 * ksq - 36 * ksq ** 2 + 24 * ksq ** 3) *
                        E2()) / 105.
            elif (q >= 4):
                d1 = q + 2 + (p + q - 2)
                d2 = (3 - q)
                res1 = (d1 * M(p, q - 2) + d2 * M(p, q - 4)) / (p + q + 3)
                d1 = (p + q - 2)
                d2 = (3 - q)
                res2 = (d1 * M(p, q - 2) + d2 * M(p, q - 4)) / (p + q + 3)
                res2 *= -ksq
                return res1 + res2
            elif (p >= 4):
                d3 = 2 * p + q - (p + q - 2)
                d4 = 0
                res1 = (d3 * M(p - 2, q)) / (p + q + 3)
                d3 = -(p + q - 2)
                d4 = (p - 3)
                res2 = (d3 * M(p - 2, q) + d4 * M(p - 4, q)) / (p + q + 3)
                res2 *= -ksq
                return res1 + res2
            else:
                raise Exception("Domain error.")

        def J(u, v):
            """Return J."""
            if (b == 0):
                return (1 - r ** 2) ** 1.5 * I(u, v)
            else:
                res = 0
                for i in range(v + 1):
                    if (is_even(i - v - u)):
                        res += choose(v, i) * M(u + 2 * i, u + 2 * v - 2 * i)
                    else:
                        res -= choose(v, i) * M(u + 2 * i, u + 2 * v - 2 * i)
                res *= 2 ** (u + 3) * (b * r) ** 1.5
                return res

        def K(u, v):
            """Return K."""
            res = 0
            for i in range(v + 1):
                res += choose(v, i) * (b / r) ** (v - i) * I(u, i)
            return res

        def L(u, v):
            """Return L."""
            res = 0
            for i in range(v + 1):
                res += choose(v, i) * (b / r) ** (v - i) * J(u, i)
            return res

        def P(l, m):
            """Return P."""
            mu = l - m
            nu = l + m
            if (is_even(nu)):
                return r ** (l + 2) * K((mu + 4) // 2, nu // 2)
            elif ((mu == 1) and is_even(l)):
                return -r ** (l - 1) * J(l - 2, 1)
            elif ((mu == 1) and not is_even(l)):
                return -r ** (l - 2) * (b * J(l - 3, 1) + r * J(l - 3, 2))
            else:
                return r ** (l - 1) * L((mu - 1) // 2, (nu - 1) // 2)

        def Q(l, m):
            """Return Q."""
            mu = l - m
            nu = l + m
            if (is_even(nu)):
                return H((mu + 4) // 2, nu // 2)
            else:
                return 0

        # Compute
        mu = l - m
        if ((l == 1) and (m == 0)):
            # I didn't bother coding this one up
            s[ind] = np.nan
        elif ((is_even(mu - 1)) and (not is_even((mu - 1) // 2))):
            s[ind] = 0
        else:
            s[ind] = Q(l, m) - P(l, m)

    return s
