"""Test exposure time integration."""
import starry2
import numpy as np


def moving_average(a, n):
    """https://stackoverflow.com/a/14314054"""
    if n % 2 != 0:
        n += 1
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    result = ret[n - 1:] / n
    return np.concatenate([np.ones(n // 2) * result[0],
                           result,
                           np.ones(n // 2 - 1) * result[-1]])


def test_exposure():
    """Test exposure time integration."""
    npts = 100000
    thin = 100

    star = starry2.kepler.Primary()
    star[1] = 0.4
    star[2] = 0.26
    planet = starry2.kepler.Secondary()
    system = starry2.kepler.System(star, planet)
    time = np.linspace(-0.01, 0.01, npts)

    # No exposure time integration
    system.exposure_time = 0
    system.compute(time, gradient=True)
    flux = system.lightcurve
    dFdt = system.gradient['time']

    # Finite exposure time w/ starry
    system.exposure_time = 0.003
    system.exposure_max_depth = 10
    system.exposure_tol = 1e-10
    system.compute(time[::thin], gradient=True)
    flux_exp = system.lightcurve
    dFdt_exp = system.gradient['time']

    # Finite exposure time: numerical
    dt = (time[-1] - time[0]) / len(time)
    n = int(system.exposure_time / dt)
    flux_num = moving_average(flux, n)
    dFdt_num = moving_average(dFdt, n)

    # Pretty lax, since the exposure time integration is not very accurate
    assert np.all(np.abs(flux_exp - flux_num[::thin]) < 1e-6)
    assert np.all(np.abs(dFdt_exp - dFdt_num[::thin]) < 1e-3)


if __name__ == "__main__":
    test_exposure()
