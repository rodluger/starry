import numpy as np
import os

np.seterr(invalid="ignore")

__all__ = ["generate_header"]


def get_f_exact(x, y, z, b):
    r"""
    Return the expression

    .. math::

        f \equiv
        \mathrm{max}\left(0, \cos(\phi_r - \phi_i)\right)
        \sin\alpha \tan\beta

    from Equation (30) in Oren & Nayar (1994) as a function
    of the Cartesian coordinates on the sky-projected sphere
    seen at a phase where the semi-minor axis of the terminator
    is `b`.

    """
    bc = np.sqrt(1 - b ** 2)
    ci = bc * y - b * z
    f1 = -b / z - ci
    f2 = -b / ci - z
    f = np.maximum(0, np.minimum(f1, f2))
    return f


def get_ijk(n):
    """Get the exponents of x, y, z i the nth term of the polynomial basis."""
    l = int(np.floor(np.sqrt(n)))
    m = n - l * l - l
    mu = l - m
    nu = l + m
    if mu % 2 == 0:
        i = mu // 2
        j = nu // 2
        k = 0
    else:
        i = (mu - 1) // 2
        j = (nu - 1) // 2
        k = 1
    return i, j, k


def poly_basis(x, y, z, deg):
    """Return the polynomial basis evaluated at `x`, `y`, `z`."""
    N = (deg + 1) ** 2
    B = np.zeros((len(x * y * z), N))
    for n in range(N):
        i, j, k = get_ijk(n)
        B[:, n] = x ** i * y ** j * z ** k
    return B


def design_matrix(x, y, z, b, deg, Nb):
    """
    Return the x-y-z-b-bc Vandermonde design matrix.
    
    NOTE: The lowest power of `b` is *ONE*, since
    we need `f = 0` eveywhere when `b = 0` for
    a smooth transition to Lambertian at crescent
    phase.
    """
    N = (deg + 1) ** 2
    u = 0
    X = np.zeros((len(y * z * b), N * Nb ** 2))
    bc = np.sqrt(1 - b ** 2)
    B = poly_basis(x, y, z, deg)
    for n in range(N):
        for p in range(1, Nb + 1):
            for q in range(Nb):
                X[:, u] = B[:, n] * b ** p * bc ** q
                u += 1
    return X


def index_of(i, j, k, p, q, deg, Nb):
    """
    Return the index in `w` corresponding to a certain term.
    
    Not at all optimized! In fact, very much the opposite.

    """
    idx = 0
    for n in range((deg + 1) ** 2):
        i0, j0, k0 = get_ijk(n)
        for p0 in range(1, Nb + 1):
            for q0 in range(Nb):
                if (
                    (i0 == i)
                    and (j0 == j)
                    and (k0 == k)
                    and (p0 == p)
                    and (q0 == q)
                ):
                    return idx
                idx += 1
    raise IndexError("Invalid polynomial index!")


def get_w(deg=5, Nb=4, res=100, prior_var=1e2):
    """
    Return the coefficients of the 5D fit to `f`
    in `x`, `y`, `z`, `b`, and `bc`.

    We fit the function `f` (see above) with a 
    polynomial of order `deg0 = deg - 1` in `x`, `y`, and `z`
    and `Nb0 = Nb - 1` in `b` and `bc`. We then multiply the
    result by `cos(theta_i)`, which is just a polynomial of
    order 1 in `y`, `z`, `b`, and `bc`.

    The final polynomial has order `deg` and `Nb` in the
    Cartesian and terminator coordinates, respectively, and is a fit
    to the function

    .. math::

        \cos\theta_i
        \mathrm{max}\left(0, \cos(\phi_r - \phi_i)\right)
        \sin\alpha \tan\beta

    from Equation (30) in Oren & Nayar (1994).

    """
    # Degrees before multiplying by cos(theta_i)
    deg0 = deg - 1
    Nb0 = Nb - 1

    # Construct a 3D grid in (x, y, b)
    bgrid = np.linspace(-1, 0, res)
    xygrid = np.linspace(-1, 1, res)
    x, y, b = np.meshgrid(xygrid, xygrid, bgrid)
    z = np.sqrt(1 - x ** 2 - y ** 2)
    idx = np.isfinite(z) & (y > b * np.sqrt(1 - x ** 2))
    x = x[idx]
    y = y[idx]
    z = z[idx]
    b = b[idx]

    # Compute the exact `f` function on this grid
    f = get_f_exact(x, y, z, b)

    # Construct the design matrix for fitting
    X = design_matrix(x, y, z, b, deg=deg0, Nb=Nb0)

    # "Data" inverse covariance. Make the errorbars large
    # when cos(theta_i) is small, since the intensity there
    # is small anyways.
    cinv = np.ones_like(f)
    bc = np.sqrt(1 - b ** 2)
    ci = bc * y - b * z
    cinv *= ci ** 2

    # Prior inverse covariance. Make odd powers of x have
    # *very* narrow priors centered on zero, since the
    # function should be symmetric about the y axis.
    # We'll explicitly zero out these coefficients below.
    PInv = np.eye((deg0 + 1) ** 2 * Nb0 ** 2)
    u = 0
    for n in range((deg0 + 1) ** 2):
        i, _, _ = get_ijk(n)
        if (i % 2) != 0:
            inv_var = 1e15
        else:
            inv_var = 1 / prior_var
        for p in range(1, Nb0 + 1):
            for q in range(Nb0):
                PInv[u, u] = inv_var
                u += 1

    # Solve the L2 problem
    XTCInv = X.T * cinv
    w0 = np.linalg.solve(XTCInv.dot(X) + PInv, XTCInv.dot(f),)

    # Zero out really tiny values.
    w0[np.abs(w0) < 1e-10] = 0.0

    # Now multiply by cos_thetai = bc * y - b * z.
    # The powers of x, y, z, b, bc are
    # i, j, k, p, q, respectively
    w = np.zeros((deg + 1) ** 2 * Nb ** 2)
    for n in range((deg0 + 1) ** 2):
        i, j, k = get_ijk(n)
        for p in range(1, Nb0 + 1):
            for q in range(Nb0):
                idx = index_of(i, j, k, p, q, deg0, Nb0)
                w[index_of(i, j + 1, k, p, q + 1, deg, Nb)] += w0[idx]

                if k == 0:
                    w[index_of(i, j, k + 1, p + 1, q, deg, Nb)] -= w0[idx]
                else:
                    # transform z^2 --> 1 - x^2 - y^2
                    w[index_of(i, j, 0, p + 1, q, deg, Nb)] -= w0[idx]
                    w[index_of(i + 2, j, 0, p + 1, q, deg, Nb)] += w0[idx]
                    w[index_of(i, j + 2, 0, p + 1, q, deg, Nb)] += w0[idx]
    return w


def generate_header(deg=5, Nb=4, res=100, nperline=3):
    print("Generating `oren_nayar.h`...")
    assert deg >= 1, "deg must be >= 1"
    assert Nb >= 1, "Nb must be >= 1"
    w = get_w(deg=deg, Nb=Nb, res=res)
    N = len(w) - (len(w) % nperline)
    string = "static const double STARRY_OREN_NAYAR_COEFFS[] = {\n"
    lines = w[:N].reshape(-1, nperline)
    last_line = w[N:]
    for i, line in enumerate(lines):
        string += ", ".join(["{:22.15e}".format(value) for value in line])
        if (i < len(lines) - 1) or (len(last_line)):
            string += ","
        string += "\n"
    string += ", ".join(["{:22.15e}".format(value) for value in last_line])
    string += "};"
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "oren_nayar.h"
        ),
        "w",
    ) as f:
        print("#ifndef _STARRY_REFLECTED_OREN_NAYAR_H_", file=f)
        print("#define _STARRY_REFLECTED_OREN_NAYAR_H_", file=f)
        print("", file=f)
        print("#define STARRY_OREN_NAYAR_DEG {}".format(deg), file=f)
        print("#define STARRY_OREN_NAYAR_N {}".format((deg + 1) ** 2), file=f)
        print("#define STARRY_OREN_NAYAR_NB {}".format(Nb), file=f)
        print("", file=f)
        print(string, file=f)
        print("#endif", file=f)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Check the quality of the fit for various values of `b`.

    # Get the polynomial coefficients
    deg = 5
    Nb = 4
    w = get_w(deg=deg, Nb=Nb)

    # Grid the surface
    res = 300
    xygrid = np.linspace(-1, 1, res)
    x, y = np.meshgrid(xygrid, xygrid)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = np.sqrt(1 - x ** 2 - y ** 2)

    # Compare for several values of b
    nimg = 6
    fig, ax = plt.subplots(nimg, 4, figsize=(6, 8))
    for axis in ax.flatten():
        axis.axis("off")
    for i, b in enumerate(np.linspace(-1, 0, nimg, endpoint=False)):

        # Illumination profile
        bc = np.sqrt(1 - b ** 2)
        ci = bc * y - b * z

        # Compute the exact `f` function on this grid
        f = ci * get_f_exact(x, y, z, b)

        # Get our approximation
        X = design_matrix(x, y, z, b, deg=deg, Nb=Nb)
        fapprox = X.dot(w)

        # Mask the nightside & reshape
        idx = np.isfinite(z) & (y < b * np.sqrt(1 - x ** 2))
        f[idx] = 0
        fapprox[idx] = 0
        f = f.reshape(res, res)
        fapprox = fapprox.reshape(res, res)

        # Plot
        vmin = 0
        vmax = 1
        ax[i, 0].imshow(
            f, origin="lower", extent=(-1, 1, -1, 1), vmin=vmin, vmax=vmax
        )
        ax[i, 1].imshow(
            fapprox,
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=vmin,
            vmax=vmax,
        )
        ax[i, 2].imshow(
            f - fapprox,
            origin="lower",
            extent=(-1, 1, -1, 1),
            vmin=-0.1,
            vmax=0.1,
            cmap="RdBu",
        )

        bins = np.linspace(-0.1, 0.1, 50)
        ax[i, 3].hist((f - fapprox).flatten(), bins=bins)
        ax[i, 3].set_xlim(-0.1, 0.1)

    plt.show()
