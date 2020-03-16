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
    xo = np.linspace(xo - 2, xo + 2, 10000)

    # The instability shows up at ydeg ~ 5 and gets bad at ydeg ~ 6
    map = starry.Map(ydeg=6, reflected=True)
    map[6, :] = 1
    flux1 = map.flux(xo=xo, yo=yo, zo=zo, ro=ro, xs=xs, ys=ys, zs=zs)

    # If `noon=True`, the flux above should be *exactly* equal to the flux of a
    # linearly-limb darkened source with u_1 = 1.0, since linear
    # limb darkening weights the surface brightness by the same
    # cosine-like profile.
    map = starry.Map(ydeg=6, udeg=1)
    map[6, :] = 1
    map[1] = 1.0
    flux2 = map.flux(xo=xo, yo=yo, zo=zo, ro=ro)

    # Plot it
    if plot:
        plt.plot(xo, flux1)
        plt.plot(xo, flux2)
        plt.ylim(-0.05, 1.5)
        plt.show()

    # TODO: Test against numerical integration
    if noon:
        assert np.allclose(flux1, flux2)
    else:
        pass


if __name__ == "__main__":
    starry.config.lazy = False
    test_I_stability(False, plot=True)
