import starry
import numpy as np
import matplotlib.pyplot as plt


def test_I_stability(plot=False):
    # FB found this unstable limit; I believe the instabilities
    # come from the upward recursion in the I integral.
    xo, yo, zo, ro = (
        31.03953239062832,
        23.892679948795926,
        1.0,
        39.10406741663172,
    )
    xo = np.linspace(xo - 2, xo + 2, 1000)

    # The instability shows up at ydeg ~ 5 and gets bad at ydeg ~ 6
    map = starry.Map(ydeg=6, reflected=True)
    map[6, :] = 1
    flux1 = map.flux(xo=xo, yo=yo, zo=zo, ro=ro, xs=0, ys=0, zs=1)

    # The flux above should be *exactly* equal to the flux of a
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

    assert np.allclose(flux1, flux2)


if __name__ == "__main__":
    starry.config.lazy = False
    test_I_stability(plot=True)
