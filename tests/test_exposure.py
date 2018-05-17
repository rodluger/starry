"""Test integration of the flux over the exposure window."""
from starry import Star, Planet, System
import numpy as np


def test_exposure():
    """Test integration of the flux over an exposure."""
    # Time arrays
    npts = 100
    time = np.linspace(-0.25, 0.25, npts)

    # Manually integrate w/ this many points (clearly ridiculous!)
    nexppts = 10000

    # Limb-darkened star
    star = Star()
    star.map[1] = 0.4
    star.map[2] = 0.26

    # Hot jupiter
    planet = Planet(r=0.01, a=60, inc=89.5, porb=50)

    # Instantiate the system
    system = System([star, planet], exposure_time=0)

    # Compute the flux with different exposure times
    for exptime in [0.001, 0.01, 0.1]:

        # Integrate manually
        flux_manual = np.zeros_like(time)
        for i, t in enumerate(time):
            system.exposure_time = 0
            system.compute(np.linspace(t - exptime / 2,
                                       t + exptime / 2, nexppts))
            flux_manual[i] = np.mean(system.flux)

        # Compute using starry's integrator
        system.exposure_time = exptime
        system.exposure_tol = 1e-8
        system.exposure_max_depth = 4
        system.compute(time)
        flux_starry = system.flux

        assert np.allclose(system.flux, star.flux + planet.flux)
        assert np.max(np.abs(flux_manual - flux_starry)) < 1e-6


if __name__ == "__main__":
    test_exposure()
