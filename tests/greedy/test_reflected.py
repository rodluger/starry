# -*- coding: utf-8 -*-
"""
Test reflected light calculations.

"""
import starry
import numpy as np
import matplotlib.pyplot as plt


def test_one_over_r_squared(n_tests=10, plot=False):
    """Test that the flux decreases as 1/r^2."""
    map = starry.Map(2, reflected=True)
    flux0 = map.flux()
    zs = np.linspace(1, 10, 100)
    flux = map.flux(xs=0, ys=0, zs=zs)

    if plot:
        plt.plot(zs, flux)
        plt.plot(zs, flux0 / zs ** 2)
        plt.show()

    assert np.allclose(flux, flux0 / zs ** 2)


def test_normalization(d=10, r=1):
    """Test the normalization of the flux."""
    # Instantiate a system. Planet has radius `r` and is at
    # distance `d` from a point illumination source.
    planet = starry.Secondary(starry.Map(reflected=True), a=d, r=r)
    star = starry.Primary(starry.Map(), r=0)
    sys = starry.System(star, planet)

    # Get the star & planet flux when it's at full phase
    t_full = 0.5 * sys._get_periods()[0]
    f_star, f_planet = sys.flux(t=t_full, total=False)

    # Star should have unit flux
    assert np.allclose(f_star, 1.0)

    # Planet should have flux equal to (2 / 3) r^2 / d^2
    assert np.allclose(f_planet, (2.0 / 3.0) * r ** 2 / d ** 2)


if __name__ == "__main__":
    starry.config.lazy = False
    test_normalization()
