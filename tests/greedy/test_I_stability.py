import starry
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize("noon", [False, True])
def test_I_stability(noon, plot=False):
    # FB found this unstable limit. The instability comes
    # from two places:
    # 1. The upward recursion in the I integral is unstable.
    #    We need to either implement a tridiagonal solver as
    #    in the J integral and/or refine our computation of kappa.
    # 2. The terminator parameter b = 1, which causes `get_angles` to
    #    oscillate between different integration codes. When b
    #    approaches unity, we should switch to the regular starry
    #    solver, since the terminator is so close to the limb that its
    #    presence doesn't matter.
    xo, yo, zo, ro = (
        31.03953239062832,
        23.892679948795926,
        1.0,
        39.10406741663172,
    )
    # Exactly noon?
    if noon:
        xs = 0.0
    else:
        xs = 0.1
    ys = 0.0
    zs = 1.0
    xo = np.linspace(xo - 2, xo + 2, 1000)

    # Compute analytic
    map = starry.Map(ydeg=10, reflected=True)
    map[10, :] = 1
    flux1 = map.flux(xo=xo, yo=yo, zo=zo, ro=ro, xs=xs, ys=ys, zs=zs)

    if noon:
        # The flux above should be *exactly* equal to 2/3 the flux of a
        # linearly-limb darkened source with u_1 = 1.0, since linear
        # limb darkening weights the surface brightness by the same
        # cosine-like profile (2/3 is the geometrical albedo of a
        # perfect Lambert sphere)
        map_e = starry.Map(ydeg=10, udeg=1)
        map_e[10, :] = 1
        map_e[1] = 1
        flux2 = (2.0 / 3.0) * map_e.flux(xo=xo, yo=yo, zo=zo, ro=ro)
        atol = 1e-12

    else:

        # Compute numerical
        res = 500
        (lat, lon), (x, y, z) = map.ops.compute_ortho_grid(res)
        image = map.render(xs=xs, ys=ys, zs=zs, res=res).flatten()
        flux2 = np.zeros_like(flux1)
        for k in range(len(xo)):
            idx = (x - xo[k]) ** 2 + (y - yo) ** 2 > ro ** 2
            flux2[k] = np.nansum(image[idx])
        flux2 *= 4 / res ** 2
        atol = 1e-3

    # Plot it
    if plot:
        plt.plot(xo, flux1)
        plt.plot(xo, flux2)
        plt.show()

    # Compare
    assert np.allclose(flux1, flux2, atol=atol)


if __name__ == "__main__":
    starry.config.lazy = False
    # test_I_stability(False, plot=True)
    test_I_stability(True, plot=True)
