from ._c_ops import sT, dsT, angles
import numpy as np
from scipy.integrate import quad
import mpmath
from mpmath import ellipe, ellipf, findroot, mpf, mpc
from scipy.special import comb, hyp2f1
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse


__all__ = [
    "PythonSolver",
    "NumericalSolver",
    "CppSolver",
    "BruteSolver",
    "draw",
]


def quartic_coeffs(f, xo, yo, ro):
    b = 1 - f
    A = (1 - b ** 2) ** 2
    B = -4 * xo * (1 - b ** 2)
    C = -2 * (
        b ** 4
        + ro ** 2
        - 3 * xo ** 2
        - yo ** 2
        - b ** 2 * (1 + ro ** 2 - xo ** 2 + yo ** 2)
    )
    D = -4 * xo * (b ** 2 - ro ** 2 + xo ** 2 + yo ** 2)
    E = (
        b ** 4
        - 2 * b ** 2 * (ro ** 2 - xo ** 2 + yo ** 2)
        + (ro ** 2 - xo ** 2 - yo ** 2) ** 2
    )
    return [A, B, C, D, E]


def find_intersections(bo, ro, f, theta):
    coeff = np.array(
        quartic_coeffs(f, bo * np.sin(theta), bo * np.cos(theta), ro)
    )
    return np.roots(coeff), np.polyval(np.abs(coeff), np.abs(np.roots(coeff)))


def circle_err(x, y, bo, ro, theta):
    return np.abs(
        (y - bo * np.cos(theta)) ** 2
        - (ro ** 2 - (x - bo * np.sin(theta)) ** 2)
    )


def get_angles(bo, ro, f, theta):

    # HACK to avoid instabilities
    if f < 1e-10:
        f = 0

    # Domain adjustment
    if bo < 0:
        bo = -bo
        theta -= np.pi

    # HACK to avoid singular point (python impl. only!)
    if np.abs(theta - 0.5 * np.pi) < 1e-3:
        theta = 0.5 * np.pi + 1e-3
    elif np.abs(theta - 1.5 * np.pi) < 1e-3:
        theta = 1.5 * np.pi + 1e-3

    # Get roots of quartic
    roots, err = find_intersections(bo, ro, f, theta)

    # Polish the roots with 30 decimal places of precision
    try:
        polished_roots = []
        with mpmath.workdps(30):
            xo = bo * mpmath.sin(theta)
            yo = bo * mpmath.cos(theta)
            A, B, C, D, E = quartic_coeffs(mpf(f), xo, yo, mpf(ro))
            poly = lambda x: A * x ** 4 + B * x ** 3 + C * x ** 2 + D * x + E
            for i, root in enumerate(roots):
                polished_root = findroot(poly, root)
                if np.abs(float(polished_root.imag)) < 1e-15:
                    polished_roots += [float(polished_root.real)]
        roots = np.array(polished_roots)
    except:
        pass

    # Center of occultor in this frame
    xo = bo * np.sin(theta)
    yo = bo * np.cos(theta)

    # Compute the corresponding angles
    phi = []
    xi = []
    if len(roots) == 2:

        # Circle intersects the edge of the ellipse twice
        for x_root in roots:
            x0, y0 = bo * np.sin(theta), bo * np.cos(theta)
            y_int = (1 - f) * np.sqrt(1 - x_root ** 2)
            phi_pos = theta + np.arctan2(y_int - y0, x_root - x0)
            xi_pos = np.arctan2(np.sqrt(1 - x_root ** 2), x_root)
            phi_neg = theta - np.arctan2(y_int + y0, x_root - x0)
            xi_neg = np.arctan2(-np.sqrt(1 - x_root ** 2), x_root)
            if circle_err(x_root, y_int, bo, ro, theta) > circle_err(
                x_root, -y_int, bo, ro, theta
            ):
                phi.append(phi_neg)
                xi.append(xi_neg)
            else:
                phi.append(phi_pos)
                xi.append(xi_pos)

        # Wrap the angles
        phi = np.array(phi) % (2 * np.pi)
        xi = np.array(xi) % (2 * np.pi)

        # xi is always counter-clockwise
        if xi[0] > xi[1]:
            xi = xi[::-1]
            phi = phi[::-1]

        # Ensure the T integral takes us
        # through the inside of the occultor
        xm = np.cos(xi.mean())
        ym = (1 - f) * np.sin(xi.mean())
        if (xm - xo) ** 2 + (ym - yo) ** 2 >= ro ** 2:
            xi = xi[::-1]
            xi[1] += 2 * np.pi

        # Ensure the P integral takes us through
        # the inside of the star
        xm = xo + ro * np.cos(theta - phi.mean())
        ym = yo - ro * np.sin(theta - phi.mean())
        if ym ** 2 > (1 - f) ** 2 * (1 - xm ** 2):
            if phi[0] < phi[1]:
                phi[0] += 2 * np.pi
            else:
                phi[1] += 2 * np.pi

        # phi is always counter-clockwise
        if phi[1] < phi[0]:
            phi = phi[::-1]

    elif len(roots) < 2:

        # Is the center of the circle outside the ellipse?
        if (np.abs(xo) > 1) or abs(yo) > (1 - f) * np.sqrt(1 - xo * xo):

            # Is the center of the ellipse outside the circle?
            if bo > ro:

                # No occultation
                phi = np.array([0.0, 0.0])
                xi = np.array([0.0, 0.0])

            else:

                # Complete occultation
                phi = np.array([0.0, 0.0])
                xi = np.array([0.0, 2 * np.pi])

        elif bo <= ro - 1:

            # Complete occultation
            phi = np.array([0.0, 0.0])
            xi = np.array([0.0, 2 * np.pi])

        else:

            # Regular occultation, but occultor doesn't touch the limb
            phi = np.array([0.0, 2 * np.pi])
            xi = np.array([0.0, 0.0])

    else:

        raise ValueError("Unexpected branch.")

    return phi, xi


def G(n, f):

    # Get the mu, nu indices
    l = int(np.floor(np.sqrt(n)))
    m = n - l * l - l
    mu = l - m
    nu = l + m

    # NOTE: The abs prevents NaNs when the argument of the sqrt is
    # zero but floating point error causes it to be ~ -eps.
    z = lambda x, y: np.maximum(
        1e-12, np.sqrt(np.abs(1 - x ** 2 - (y / (1 - f)) ** 2))
    )

    if nu % 2 == 0:

        G = [
            lambda x, y: 0,
            lambda x, y: x ** (0.5 * (mu + 2)) * (y / (1 - f)) ** (0.5 * nu),
        ]

    elif (l == 1) and (m == 0):

        def G0(x, y):
            z_ = z(x, y)
            if z_ > 1 - 1e-8:
                return -0.5 * y
            else:
                return (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * (-y)

        def G1(x, y):
            z_ = z(x, y)
            if z_ > 1 - 1e-8:
                return 0.5 * x
            else:
                return (1 - z_ ** 3) / (3 * (1 - z_ ** 2)) * x

        G = [G0, G1]

    elif (mu == 1) and (l % 2 == 0):

        G = [
            lambda x, y: (1 - f) * x ** (l - 2) * z(x, y) ** 3,
            lambda x, y: 0,
        ]

    elif (mu == 1) and (l % 2 != 0):

        G = [lambda x, y: x ** (l - 3) * y * z(x, y) ** 3, lambda x, y: 0]

    else:

        G = [
            lambda x, y: 0,
            lambda x, y: x ** (0.5 * (mu - 3))
            * (y / (1 - f)) ** (0.5 * (nu - 1))
            * z(x, y) ** 3,
        ]

    return G


def primitive(x, y, dx, dy, phi1, phi2, f, theta, n=0):
    def func(phi):
        Gx, Gy = G(n, f)
        xr = lambda phi: x(phi) * np.cos(theta) + y(phi) * np.sin(theta)
        yr = lambda phi: -x(phi) * np.sin(theta) + y(phi) * np.cos(theta)
        dxr = lambda phi: dx(phi) * np.cos(theta) + dy(phi) * np.sin(theta)
        dyr = lambda phi: -dx(phi) * np.sin(theta) + dy(phi) * np.cos(theta)
        return Gx(xr(phi), yr(phi)) * dxr(phi) + Gy(xr(phi), yr(phi)) * dyr(
            phi
        )

    res, _ = quad(func, phi1, phi2, epsabs=1e-12, epsrel=1e-12)
    return res


def pTbar_numerical(lmax, phi, b, r, f, theta, linear=False):
    pT = np.zeros((lmax + 1) ** 2)
    for n in range((lmax + 1) ** 2):
        for phi1, phi2 in np.reshape(phi, (-1, 2)):
            l = int(np.floor(np.sqrt(n)))
            m = n - l * l - l
            mu = l - m
            nu = l + m
            if (not linear) or (nu % 2 == 0) or (l == 1 and m == 0):
                # Exact integral
                x = lambda phi: r * np.cos(phi)
                y = lambda phi: (b + r * np.sin(phi))
                dx = lambda phi: -r * np.sin(phi)
                dy = lambda phi: r * np.cos(phi)
                pT[n] += primitive(x, y, dx, dy, phi1, phi2, f, theta, n)
            else:
                # Linear approximation in `f`
                x = lambda phi: r * np.cos(theta - phi) + b * np.sin(theta)
                y = lambda phi: b * np.cos(theta) - r * np.sin(theta - phi)
                z = lambda phi: np.sqrt(np.abs(1 - x(phi) ** 2 - y(phi) ** 2))
                dz3df = lambda phi: -3 * y(phi) ** 2 * z(phi)
                if (mu == 1) and (l % 2 == 0):
                    func = lambda phi: (
                        r
                        * (1 - f)
                        * x(phi) ** (l - 2)
                        * (z(phi) ** 3 + dz3df(phi) * f)
                        * np.sin(theta - phi)
                    )
                elif (mu == 1) and (l % 2 != 0):
                    func = lambda phi: (
                        r
                        * x(phi) ** (l - 3)
                        * y(phi)
                        * (z(phi) ** 3 + dz3df(phi) * f)
                        * np.sin(theta - phi)
                    )
                else:
                    func = lambda phi: (
                        r
                        * (1 - f) ** ((1 - nu) // 2)
                        * x(phi) ** ((mu - 3) // 2)
                        * y(phi) ** ((nu - 1) // 2)
                        * (z(phi) ** 3 + dz3df(phi) * f)
                        * np.cos(theta - phi)
                    )
                res, _ = quad(func, phi1, phi2, epsabs=1e-12, epsrel=1e-12)
                pT[n] += res

    return pT


def tTbar_numerical(lmax, xi, b, r, f, theta):
    tT = np.zeros((lmax + 1) ** 2)
    for n in range((lmax + 1) ** 2):
        for xi1, xi2 in np.reshape(xi, (-1, 2)):
            x = lambda xi: np.cos(theta) * np.cos(xi) - (1 - f) * np.sin(
                theta
            ) * np.sin(xi)
            y = lambda xi: np.sin(theta) * np.cos(xi) + (1 - f) * np.cos(
                theta
            ) * np.sin(xi)
            dx = lambda xi: -np.cos(theta) * np.sin(xi) - (1 - f) * np.sin(
                theta
            ) * np.cos(xi)
            dy = lambda xi: -np.sin(theta) * np.sin(xi) + (1 - f) * np.cos(
                theta
            ) * np.cos(xi)
            tT[n] += primitive(x, y, dx, dy, xi1, xi2, f, theta, n)
    return tT


def sTbar_numerical(lmax, b, r, f, theta, linear=False):
    phi, xi = get_angles(b, r, f, theta)
    return pTbar_numerical(
        lmax, phi, b, r, f, theta, linear=linear
    ) + tTbar_numerical(lmax, xi, b, r, f, theta)


def integrate(f, a, b, epsabs=1e-12, epsrel=1e-12, **kwargs):
    return quad(f, a, b, epsabs=epsabs, epsrel=epsrel, **kwargs)[0]


def get_sT0(lmax):
    # Compute the solution vector for no occultation
    L = np.zeros((lmax + 3, lmax + 3))
    L[0, 0] = 2 * np.pi
    for u in range(0, lmax + 3, 2):
        for v in range(2, lmax + 3, 2):
            fac = (v - 1.0) / (u + v)
            L[u, v] = fac * L[u, v - 2]
            L[v, u] = fac * L[v - 2, u]
    sT0 = np.zeros((lmax + 1) ** 2)
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m
            if nu % 2 == 0:
                # Case 1
                sT0[n] = L[mu // 2 + 2, nu // 2]
            elif (l == 1) and (m == 0):
                # Case 2
                sT0[n] = 2 * np.pi / 3
            n += 1
    return sT0


class PythonSolver:
    def __init__(self, lmax=5):

        # Pre-compute case conditions
        self.lmax = lmax
        self.cases = np.zeros((self.lmax + 1) ** 2, dtype=int)
        self.l = np.zeros((self.lmax + 1) ** 2, dtype=int)
        self.m = np.zeros((self.lmax + 1) ** 2, dtype=int)
        for n in range((self.lmax + 1) ** 2):
            self.cases[n] = self.get_case(n)
            self.l[n] = int(np.floor(np.sqrt(n)))
            self.m[n] = n - self.l[n] ** 2 - self.l[n]
        self.sT0 = get_sT0(self.lmax)

    def get_case(self, n):
        l = int(np.floor(np.sqrt(n)))
        m = n - l ** 2 - l
        mu = l - m
        nu = l + m
        if (nu % 2) == 0:
            return 1
        elif (l == 1) and (m == 0):
            return 2
        elif ((nu % 2) == 1) and (mu == 1) and ((l % 2) == 0):
            return 3
        elif ((nu % 2) == 1) and (mu == 1) and ((l % 2) == 1):
            return 4
        else:
            return 5

    def solve(self, f0, fN, A, B, C, N):
        """
        Solve a recursion relation of the form
        
            f(j) = A(j) * f(j - 1) + B(j) * f (j - 2) + C(j)
        
        given a lower boundary condition f(0) and 
        an upper boundary condition f(N).
        
        """
        # Set up the tridiagonal problem
        a = np.empty(N - 1)
        b = np.empty(N - 1)
        c = np.empty(N - 1)
        for i, j in enumerate(range(2, N + 1)):
            a[i] = -A(j)
            b[i] = -B(j)
            c[i] = C(j)

        # Add the boundary conditions
        c[0] -= b[0] * f0
        c[-1] -= fN

        # Construct the tridiagonal matrix
        A = np.diag(a, 0) + np.diag(b[1:], -1) + np.diag(np.ones(N - 2), 1)

        # Solve using a dense solver for stability
        soln = np.linalg.solve(A, c)
        return np.concatenate(([f0], soln, [fN]))

    def get_W(self, k2, phi1, phi2):
        """
        Return the vector of `W` integrals, computed
        recursively given a lower boundary condition
        (analytic in terms of elliptic integrals) and an upper
        boundary condition (computed numerically).

        The term `W_j` is the solution to the integral of
        
            sin(u)^(2 * j) * sqrt(1 - sin(u)^2 / (1 - k^2))

        from u = u1 to u = u2, where 
        
            u = (pi - 2 * phi) / 4

        """
        # Useful quantities
        N = 2 * self.lmax + 4
        kc2 = 1 - k2
        u1 = 0.25 * (np.pi - 2 * phi1)
        u2 = 0.25 * (np.pi - 2 * phi2)
        u = np.array([u1, u2])
        sinu = np.sin(u)
        cosu = np.cos(u)
        diff = lambda x: np.diff(x)[0]
        D2 = 1 - sinu ** 2 / kc2
        if k2 >= 1:
            sgn = 1
        else:
            sgn = -1
        D3 = sgn * abs(D2) ** 1.5

        # The two boundary conditions
        if sgn == 1:
            f0 = float((ellipe(u[1], 1 / kc2) - ellipe(u[0], 1 / kc2)).real)
        else:
            f0 = float((ellipe(u[1], 1 / kc2) - ellipe(u[0], 1 / kc2)).imag)
            """
            For reference, this expression can be written in terms of the
            real parts of elliptic integrals as follows:

                n = np.floor((u + 0.5 * np.pi) / np.pi)
                term = np.sqrt((sinu ** 2 / kc2 - 1) * (1 / k2 - 1))
                A = n * np.pi + (-1) ** n * np.arcsin(term / sinu)
                F = float((ellipf(A[1], k2) - ellipf(A[0], k2)).real)
                E = float((ellipe(A[1], k2) - ellipe(A[0], k2)).real)
                C = diff(
                    (np.sin(A) * np.cos(A)) / np.sqrt(1 - k2 * np.sin(A) ** 2)
                )
                f0 = -np.sqrt(kc2) * (F - E / kc2 + k2 / kc2 * C)
            """

        fN = quad(
            lambda u: np.sin(u) ** (2 * N)
            * abs(1 - np.sin(u) ** 2 / kc2) ** 0.5,
            u1,
            u2,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]

        # The recursion coefficients
        A = lambda j: 2 * (j + (j - 1) * kc2) / (2 * j + 1)
        B = lambda j: -(2 * j - 3) / (2 * j + 1) * kc2
        C = lambda j: diff(kc2 * sinu ** (2 * j - 3) * cosu * D3) / (2 * j + 1)

        # Solve the linear system
        return self.solve(f0, fN, A, B, C, N)

    def get_V(self, k2, phi1, phi2):
        """
        Return the vector of `V` integrals, computed
        recursively given a lower boundary condition
        (trivial) and an upper boundary condition 
        (computed from `2F1`). 
        
        The term `V_i` is the solution to the integral of
        
            cos(phi) sin(phi)^i sqrt(1 - w^2 sin(phi))
        
        from phi=phi1 to phi=phi2.
        
        """
        N = 2 * self.lmax + 4
        invw2 = 2 * k2 - 1
        if k2 < 0.5:
            sgn = -1
        else:
            sgn = 1

        V = np.zeros((N + 1, 2))
        for i, phi in enumerate([phi1, phi2]):

            # Useful quantities
            sinphi = np.sin(phi)
            x = sinphi / invw2

            # The two boundary conditions
            if sgn == 1:
                f0 = (2.0 / 3.0) * (1 - (1 - x) ** 1.5) * invw2
                fN = (
                    sinphi ** (N + 1) / (N + 1) * hyp2f1(-0.5, N + 1, N + 2, x)
                )
            else:
                # NOTE: These are the original expressions:
                #
                # f0 = ((2.0 / 3.0) * (1 - (1 - x + 0j) ** 1.5) * invw2).imag
                # fN = -(
                #    sinphi ** (N + 1)
                #    / (N + 1)
                #    * hyp2f1(-0.5, N + 1, N + 2, x + 0j)
                # ).imag
                #
                term = (2.0 / 3.0) * (x - 1) ** 1.5
                f0 = term * invw2
                # This is equivalent:
                # fN = f0 * invw2 ** N * hyp2f1(1.5, -N, 2.5, 1 - x)
                fN = (
                    term * sinphi ** (N + 1) * hyp2f1(1.0, N + 2.5, 2.5, 1 - x)
                )

            # The recursion coefficients
            A = lambda j: (2 * j + (2 * j + 1) * x) * invw2 / (2 * j + 3)
            B = lambda j: -2 * (j - 1) * sinphi * invw2 / (2 * j + 3)
            C = lambda j: 0.0

            # Solve the linear system
            V[:, i] = self.solve(f0, fN, A, B, C, N)

        return np.diff(V, axis=1)

    def get_J(self, k2, phi1, phi2):
        """
        Return the `J` matrix of integrals. The term `J_{p,q}`
        is equal to the integral of
        
            cos(phi)^p sin(phi)^q sqrt(1 - w^2 sin(phi))
            
        from phi=phi1 to phi=phi2.
        
        """
        W = self.get_W(k2, phi1, phi2)
        V = self.get_V(k2, phi1, phi2)
        w2 = 1 / (2 * k2 - 1)
        c1 = -2.0 * abs(1 - w2) ** 0.5
        J = np.zeros((self.lmax + 3, self.lmax + 3))
        for s in range((self.lmax + 3) // 2):
            for q in range(self.lmax + 3):
                fac = 1
                for i in range(s + 1):
                    term = 0
                    amp = 1
                    for j in range(2 * i + q + 1):
                        term += amp * W[j]
                        amp *= -2 * (2 * i + q - j) / (j + 1)
                    J[2 * s, q] += c1 * fac * term
                    J[2 * s + 1, q] += fac * V[2 * i + q]
                    fac *= (i - s) / (i + 1)
        return J

    def get_L(self, phip1, phip2):
        """
        Return the `L` matrix of integrals. The term `L_{i, j}`
        is equal to the integral of
        
            cos(phi)^i sin(phi)^j
            
        from phi=phip1 to phi=phip2, where
        
            phip = phi - theta
            
        (pT integral) or
        
            phip = xi
            
        (tT integral).
        
        """
        # Lower boundary
        L = np.zeros((self.lmax + 3, self.lmax + 3))
        L[0, 0] = phip2 - phip1
        L[1, 0] = np.sin(phip2) - np.sin(phip1)
        L[0, 1] = np.cos(phip1) - np.cos(phip2)
        L[1, 1] = 0.5 * (np.cos(phip1) ** 2 - np.cos(phip2) ** 2)

        # Recursion coeffs
        cp1 = np.cos(phip1)
        cp2 = np.cos(phip2)
        sp1 = np.sin(phip1)
        sp2 = np.sin(phip2)
        A0 = cp1 * sp1
        B0 = cp2 * sp2
        C0 = cp2 * sp2
        D0 = cp1 * sp1

        # Recurse
        for u in range(self.lmax + 3):
            A = A0
            B = B0
            C = C0
            D = D0
            for v in range(2, self.lmax + 3):
                fac = 1.0 / (u + v)
                L[u, v] = fac * (A - B + (v - 1) * L[u, v - 2])
                L[v, u] = fac * (C - D + (v - 1) * L[v - 2, u])
                A *= sp1
                B *= sp2
                C *= cp2
                D *= cp1
            A0 *= cp1
            B0 *= cp2
            C0 *= sp2
            D0 *= sp1

        return L

    def get_pT2(self, r, theta, f, b, phi1, phi2):
        """
        Return the `pT[2]` integral, computed by numerical integration.
        
        """
        bc = b * np.cos(theta)
        bs = b * np.sin(theta)
        p = (1 - f) ** (-2)

        def func(phi):
            z = np.sqrt(
                1
                - (r * np.cos(phi - theta) + bs) ** 2
                - p * (r * np.sin(phi - theta) + bc) ** 2
            )
            z = max(1e-12, min(z, 1 - 1e-8))
            return (
                r * (r + b * np.sin(phi)) * (1 - z ** 3) / (3 * (1 - z ** 2))
            )

        res, _ = quad(func, phi1, phi2, epsabs=1e-12, epsrel=1e-12)
        return res

    def _get_sTbar(
        self, bo=0.58, ro=0.4, f=0.2, theta=0.5,
    ):
        """
        Return the complement of the `sT` solution vector.

        """
        # Stability hacks
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        if np.abs(bo) < 1e-15:
            bo = 1e-15
        if np.abs(sintheta) < 1e-15:
            sintheta = 1e-15
        elif np.abs(costheta) < 1e-15:
            costheta = 1e-15
        if np.abs(bo - ro) < 1e-12:
            bo = ro + 1e-12
        if np.abs(bo - ro) < 1e-5 and np.abs(ro - 0.5) < 1e-5:
            bo = ro + 1e-5
        if np.abs(1 - bo - ro) < 1e-6:
            bo = 1 - ro + 1e-6

        # Get the angles
        (phi1, phi2), (xi1, xi2) = get_angles(bo, ro, f, theta)

        # Special cases
        if (phi1 == phi2 == xi1 == 0) and (xi2 == 2 * np.pi):
            # No occultation?
            sT = np.zeros((self.lmax + 1) ** 2)
            Lxi = (1 - f) * self.get_L(xi1, xi2)
            for n in range((self.lmax + 1) ** 2):
                l = self.l[n]
                m = self.m[n]
                mu = l - m
                nu = l + m
                if self.cases[n] == 1:
                    tT = Lxi[mu // 2 + 2, nu // 2]
                    sT[n] = tT
                elif self.cases[n] == 2:
                    tT = (1 - f) * (xi2 - xi1) / 3
                    sT[n] = tT
            return sT
        elif phi1 == phi2 == xi1 == xi2 == 0:
            # Complete occultation
            return np.zeros((self.lmax + 1) ** 2)

        # Useful variables
        b = bo
        r = ro
        tantheta = sintheta / costheta
        invtantheta = 1.0 / tantheta
        gamma = 1 - b ** 2 - r ** 2
        sqrtgamma = np.sqrt(np.abs(gamma))
        w2 = 2 * b * r / gamma
        k2 = (gamma + 2 * b * r) / (4 * b * r)

        # Compute the S, C matrices
        S = np.zeros((self.lmax + 3, self.lmax + 3))
        C = np.zeros((self.lmax + 3, self.lmax + 3))
        fac0 = 1
        for i in range(self.lmax + 3):
            facs = r * fac0
            facc = fac0
            for j in range(i + 1):
                S[i, self.lmax + 2 + j - i] = facs
                C[self.lmax + 2 + j - i, i] = facc
                fac = b * (i - j) / (r * (j + 1))
                facs *= fac * sintheta
                facc *= fac * costheta
            fac0 *= r

        # Compute the B matrices
        A = np.empty((self.lmax + 3, self.lmax + 3))
        B = np.empty((self.lmax + 3, self.lmax + 3))
        fac0l = 1
        fac0r = 1
        for i in range(self.lmax + 3):
            facl = fac0l
            facr = fac0r
            for j in range(self.lmax + 3):
                A[i, j] = facl
                B[i, j] = facr
                fac = (i - j) / (j + 1)
                facl *= fac * tantheta
                facr *= -fac * invtantheta
            fac0l *= costheta
            fac0r *= -sintheta

        # Compute the first f derivative matrix
        D = -3 * np.array(
            [
                [
                    0.5
                    * (
                        b ** 2 + r ** 2 + (b + r) * (b - r) * np.cos(2 * theta)
                    ),
                    2 * b * r * costheta ** 2,
                    r ** 2 * np.cos(2 * theta),
                ],
                [
                    -2 * b * r * costheta * sintheta,
                    -2 * r ** 2 * costheta * sintheta,
                    0,
                ],
            ]
        )

        # Compute the L and M integrals
        L = self.get_L(phi1 - theta, phi2 - theta)
        Lxi = (1 - f) * self.get_L(xi1, xi2)
        M = S[:-1, 1:] @ L[::-1, ::-1][:-1, :] @ C[:, :-1]

        # Compute the J integrals
        J = self.get_J(k2, phi1, phi2)
        J32 = gamma * sqrtgamma * (J[:-1, :-1] - w2 * J[:-1, 1:])
        J12 = sqrtgamma * J[:-1, :-1]

        # Compute the I integral
        I = np.zeros((self.lmax + 2, self.lmax + 2))
        T = np.zeros((self.lmax + 1, self.lmax))
        for i in range(self.lmax + 1):
            for j in range(min(self.lmax, i + 1)):

                # Taylor series in `f`
                J0 = J32[i - j, j]
                J1 = np.sum(D * (J12[i - j :, j:])[:2, :3])
                T[i, j] = J0 + f * J1

                for k in range(i + 1):
                    imk = i - k
                    for l in range(max(0, j - imk), min(k, j) + 1):
                        I[self.lmax + 1 - imk, self.lmax + 1 - k] += (
                            A[imk, j - l] * T[i, j] * B[k, l]
                        )

        # Compute the K, H integrals
        K = S[:-1, 2:] @ I[:-1, :] @ C[1:, :-1]
        H = S[:-1, 1:] @ I[:, :-1] @ C[2:, :-1]

        # Compute sT on a case-by-case basis
        sT = np.zeros((self.lmax + 1) ** 2)
        for n in range((self.lmax + 1) ** 2):
            l = self.l[n]
            m = self.m[n]
            mu = l - m
            nu = l + m
            if self.cases[n] == 1:
                pT = (1 - f) ** (-nu // 2) * M[(mu + 2) // 2, nu // 2]
                tT = Lxi[mu // 2 + 2, nu // 2]
                sT[n] = pT + tT
            elif self.cases[n] == 2:
                pT = self.get_pT2(r, theta, f, b, phi1, phi2)
                tT = (1 - f) * (xi2 - xi1) / 3
                sT[n] = pT + tT
            elif self.cases[n] == 3:
                sT[n] = -(1 - f) * H[l - 2, 0]
            elif self.cases[n] == 4:
                sT[n] = -H[l - 3, 1]
            else:
                sT[n] = (1 - f) ** ((1 - nu) // 2) * K[
                    (mu - 3) // 2, (nu - 1) // 2
                ]
        return sT

    def get_sT(self, nruns=1, f=0.2, **kwargs):
        for n in range(nruns):
            res = (1 - f) * self.sT0 - self._get_sTbar(f=f, **kwargs)
        return res


class NumericalSolver:
    def __init__(self, lmax=5, linear=False):
        self.lmax = lmax
        self.linear = linear
        self.sT0 = get_sT0(self.lmax)

    def get_sT(self, bo=0.58, ro=0.4, f=0.2, theta=0.5, nruns=1):
        for n in range(nruns):
            res = (1 - f) * self.sT0 - sTbar_numerical(
                self.lmax, bo, ro, f, theta, linear=self.linear
            )
        return res


class CppSolver:
    def __init__(self, lmax=5):
        self.lmax = lmax

    def get_sT(
        self, bo=0.58, ro=0.4, f=0.2, theta=0.5, nruns=1,
    ):
        return sT(self.lmax, bo, ro, f, theta, nruns)


class BruteSolver:
    def __init__(self, lmax=5, res=999, grid_star=False):
        self.lmax = lmax
        self.res = res
        self.grid_star = grid_star
        self.sT0 = get_sT0(self.lmax)

    def g(self, n, x, y, z=None):
        """
        Return the nth term of the Green's basis.
        
        """
        if z is None:
            z2 = 1 - x ** 2 - y ** 2
            z = np.sqrt(np.abs(z2))
            on_star = z2 > 0
            z[~on_star] = np.nan
        l = int(np.floor(np.sqrt(n)))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            I = [mu // 2]
            J = [nu // 2]
            K = [0]
            C = [(mu + 2) // 2]
        elif (l == 1) and (m == 0):
            I = [0]
            J = [0]
            K = [1]
            C = [1]
        elif (mu == 1) and (l % 2 == 0):
            I = [l - 2]
            J = [1]
            K = [1]
            C = [3]
        elif mu == 1:
            I = [l - 3, l - 1, l - 3]
            J = [0, 0, 2]
            K = [1, 1, 1]
            C = [-1, 1, 4]
        else:
            I = [(mu - 5) // 2, (mu - 5) // 2, (mu - 1) // 2]
            J = [(nu - 1) // 2, (nu + 3) // 2, (nu - 1) // 2]
            K = [1, 1, 1]
            C = [(mu - 3) // 2, -(mu - 3) // 2, -(mu + 3) // 2]
        res = z * 0
        for i, j, k, c in zip(I, J, K, C):
            if c != 0:
                res += c * x ** i * y ** j * z ** k

        return res

    def get_sT(
        self, bo=0.58, ro=0.4, f=0.2, theta=0.5, nruns=1,
    ):
        if self.grid_star:

            # Grid up the star (accurate for large `ro`)

            for n in range(nruns):

                # Semi-minor axis of the ellipse
                b = 1 - f

                # There are two equivalent ways of doing this:
                if False:

                    # Grid up the ellipse
                    x = np.linspace(-1 - ro, 1 + ro, self.res)
                    y = np.linspace(-b - ro, b + ro, self.res)
                    x, y = np.meshgrid(x, y)
                    on_star = 1 - x ** 2 - (y / b) ** 2 > 0

                    # Compute the Green's basis in (x, y') where
                    #
                    #     y' = y / b
                    #
                    # is the transformed y coordinate
                    g = np.array(
                        [
                            self.g(n, x, y / b)
                            for n in range((self.lmax + 1) ** 2)
                        ]
                    )
                    g /= np.count_nonzero(on_star)

                    # Compute the visible pixels
                    xo = bo * np.sin(theta)
                    yo = bo * np.cos(theta)
                    under_occultor = (x - xo) ** 2 + (y - yo) ** 2 <= ro ** 2
                    inds = on_star & ~under_occultor

                else:

                    # Grid up the unit disk
                    x = np.linspace(-1 - ro, 1 + ro, self.res)
                    y = np.linspace(-1 - ro, 1 + ro, self.res)
                    x, y = np.meshgrid(x, y)
                    on_star = 1 - x ** 2 - y ** 2 > 0

                    # Compute the Green's basis as usual
                    g = np.array(
                        [self.g(n, x, y) for n in range((self.lmax + 1) ** 2)]
                    )
                    g /= np.count_nonzero(on_star)

                    # Compute the visible pixels in a stretched frame where
                    #
                    #     y' = y * b
                    #
                    # is the transformed y coordinate
                    xo = bo * np.sin(theta)
                    yo = bo * np.cos(theta)
                    under_occultor = (x - xo) ** 2 + (
                        y * b - yo
                    ) ** 2 <= ro ** 2
                    inds = on_star & ~under_occultor

            # Integrate
            sT = b * np.pi * np.sum(g[:, inds], axis=1)

            return sT

        else:

            # Grid up the planet and subtract the integral from
            # the solution over the entire disk (accurate for small `ro`)

            for n in range(nruns):

                # Semi-minor axis of the ellipse
                b = 1 - f

                # Location of the occultor
                xo = bo * np.sin(theta)
                yo = bo * np.cos(theta)

                # Grid up the occultor
                x = np.linspace(xo - ro, xo + ro, self.res)
                y = np.linspace(yo - ro, yo + ro, self.res)
                x, y = np.meshgrid(x, y)
                on_star = 1 - x ** 2 - (y / b) ** 2 > 0

                # Compute the Green's basis in (x, y') where
                #
                #     y' = y / b
                #
                # is the transformed y coordinate
                g = np.array(
                    [self.g(n, x, y / b) for n in range((self.lmax + 1) ** 2)]
                )

                # Compute the masked pixels
                under_occultor = (x - xo) ** 2 + (y - yo) ** 2 <= ro ** 2
                inds = on_star & under_occultor
                g /= np.count_nonzero(under_occultor)

            # The solution vector for the complement of the visible region
            sTbar = np.pi * ro ** 2 * np.sum(g[:, inds], axis=1)

            return b * self.sT0 - sTbar


def draw(bo=0.58, ro=0.4, f=0.2, theta=0.5, file=None):

    # Domain adjustment
    if bo < 0:
        bo = -bo
        theta -= np.pi

    # Set up the figure
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(-1.01 - 2 * ro, 1.01 + 2 * ro)
    ax.set_ylim(-1.01 - 2 * ro, 1.01 + 2 * ro)
    ax.set_aspect(1)
    ax.axis("off")

    # Draw the two bodies
    occulted = Ellipse((0, 0), 1.0 * 2, (1 - f) * 2, 0, fill=False, color="k")
    ax.add_artist(occulted)
    occulted_max = Circle(
        (0, 0), 1.0, fill=False, color="k", alpha=0.1, ls="--", lw=1
    )
    ax.add_artist(occulted_max)
    occulted_min = Circle(
        (0, 0), 1.0 - f, fill=False, color="k", alpha=0.1, ls="--", lw=1
    )
    ax.add_artist(occulted_min)
    occultor = Circle(
        (bo * np.sin(theta), bo * np.cos(theta)), ro, fill=False, color="r"
    )
    ax.add_artist(occultor)
    ax.plot(0, 0, "ko", ms=3)
    ax.plot(bo * np.sin(theta), bo * np.cos(theta), "ro", ms=3)

    # Get angles
    phi, xi = get_angles(bo, ro, f, theta)
    ax.set_title(
        r"$\phi: [{:.0f}^\circ, {:.0f}^\circ] \,\,\,\,\,\,\,\,\,\,\,\,\,\,\, \xi: [{:.0f}^\circ, {:.0f}^\circ]$".format(
            *(phi * 180 / np.pi), *(xi * 180 / np.pi)
        ),
        fontsize=12,
    )

    # Integration path along occultor (white --> red)
    x = np.zeros(10000)
    y = np.zeros(10000)
    for k, phi_k in enumerate(np.linspace(phi[0], phi[1], 10000)):
        x0 = ro * np.cos(phi_k)
        y0 = bo + ro * np.sin(phi_k)
        x[k] = x0 * np.cos(theta) + y0 * np.sin(theta)
        y[k] = -x0 * np.sin(theta) + y0 * np.cos(theta)
    plt.plot(x, y, "r", zorder=99)
    plt.plot(x[0], y[0], "ro", zorder=100, ms=5)
    plt.plot(x[-1], y[-1], "ro", zorder=100, ms=5)
    for i in [len(x) // 4, len(x) // 2, (3 * len(x)) // 4, len(x) - 2]:
        plt.annotate(
            "",
            xytext=(x[i], y[i]),
            xy=(x[i + 1], y[i + 1]),
            arrowprops=dict(arrowstyle="->", color="r"),
            size=15,
        )

    # Integration path along star (white --> black)
    x = np.zeros(10000)
    y = np.zeros(10000)
    for k, xi_k in enumerate(np.linspace(xi[0], xi[1], 10000)):
        x[k] = np.cos(xi_k)
        y[k] = (1 - f) * np.sin(xi_k)
    plt.plot(x, y, "k", zorder=99)
    plt.plot(x[0], y[0], "ko", zorder=101, ms=3)
    plt.plot(x[-1], y[-1], "ko", zorder=101, ms=3)
    for i in [len(x) // 4, len(x) // 2, (3 * len(x)) // 4, len(x) - 2]:
        plt.annotate(
            "",
            xytext=(x[i], y[i]),
            xy=(x[i + 1], y[i + 1]),
            arrowprops=dict(arrowstyle="->", color="k"),
            size=15,
        )

    # Save or display
    if file is not None:
        fig.savefig(file, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
