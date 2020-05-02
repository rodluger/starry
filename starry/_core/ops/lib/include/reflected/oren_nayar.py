import numpy as np
import os

np.seterr(invalid="ignore")

__all__ = ["generate_header"]


def get_f_exact(x, y, z, b):
    r"""
    Return the expression

    .. math::

        f \equiv
        \cos\theta_i
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
    f = ci * np.maximum(0, np.minimum(f1, f2))
    return f


def poly_basis(x, y, z, deg):
    """Return the polynomial basis evaluated at `x`, `y`, `z`."""
    N = (deg + 1) ** 2
    B = np.zeros((len(x * y * z), N))
    for n in range(N):
        l = int(np.floor(np.sqrt(n)))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            i = mu // 2
            j = nu // 2
            k = 0
        else:
            i = (mu - 1) // 2
            j = (nu - 1) // 2
            k = 1
        B[:, n] = x ** i * y ** j * z ** k
    return B


def design_matrix(x, y, z, b, deg=4, Nb=3):
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


def get_w6(
    deg=4, Nb=3, res=100, inv_var=1e-4, term_eps=1e-3, term_inv_var=1e6
):
    """
    Return the coefficients of the 5D fit to `f`
    in `x`, `y`, `z`, `b`, and `bc`.

    """
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
    X = design_matrix(x, y, z, b, deg=deg, Nb=Nb)

    # Set the "data" covariance to be diagonal, with unit
    # variance everywhere *except* very close to the terminator,
    # where we make the variance tiny. This ensures that
    # our intensity falls to (almost) zero at the terminator,
    # ensuring it's continuous across the day/night boundary.
    # There's definitely a more elegant way to do this: we could
    # force the fit to be proportional to the Lambertian term
    # `ci = bc * y - b * z`, which is zero along the terminator.
    # But this hack is likely sufficient for now.
    terminator = np.abs(y - b * np.sqrt(1 - x ** 2)) < term_eps
    cinv = np.ones_like(f)
    cinv[terminator] = term_inv_var

    # Solve the linear problem
    N = (deg + 1) ** 2
    XTCInv = X.T * cinv
    w6 = np.linalg.solve(
        XTCInv.dot(X) + inv_var * np.eye(N * Nb ** 2), XTCInv.dot(f),
    )

    return w6


def generate_header(deg=4, Nb=3, res=100, inv_var=1e-4, nperline=3):
    print("Generating `oren_nayar.h`...")
    assert deg >= 1, "deg must be >= 1"
    assert Nb >= 1, "Nb must be >= 1"
    w6 = get_w6(deg=deg, Nb=Nb, res=res, inv_var=inv_var)
    N = len(w6) - (len(w6) % nperline)
    string = "static const double STARRY_OREN_NAYAR_COEFFS[] = {\n"
    lines = w6[:N].reshape(-1, nperline)
    last_line = w6[N:]
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

    # Get the coefficients
    deg = 4
    Nb = 3
    w6 = get_w6(deg=deg, Nb=Nb)

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

        # Compute the exact `f` function on this grid
        f = get_f_exact(x, y, z, b)

        # Get our approximation
        X = design_matrix(x, y, z, b, deg=deg, Nb=Nb)
        fapprox = X.dot(w6)

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
