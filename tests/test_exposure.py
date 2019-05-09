# -*- coding: utf-8 -*-
import exoplanet as exo
import starry
import numpy as np
import pytest


def moving_average(a, n):
    """
    Compute a moving average over a window
    of `n` points. Based on
    https://stackoverflow.com/a/14314054
    """
    if n % 2 != 0:
        n += 1
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    result = ret[n - 1:] / n
    return np.concatenate([np.ones(n // 2) * result[0],
                           result,
                           np.ones(n // 2 - 1) * result[-1]])


def test_ld():
    texp = 0.05
    map = starry.Map(udeg=2)
    map[1:] = [0.4, 0.26]
    orbit = exo.orbits.KeplerianOrbit(period=1.0, m_star=1.0, r_star=1.0)
    t = np.linspace(-0.2, 0.2, 10000)
    flux = map.flux(t=t, orbit=orbit, ro=0.1).eval()
    fluence_mavg = moving_average(flux, int(texp / (t[1] - t[0])))
    fluence_starry = map.flux(t=t, orbit=orbit, ro=0.1, 
                              texp=texp, oversample=30).eval()
    fluence_starry_vec = map.flux(t=t, orbit=orbit, ro=0.1, 
                              texp=np.ones_like(t) * texp, oversample=30).eval()
    assert np.allclose(fluence_mavg, fluence_starry, fluence_starry_vec)


def test_ylm_occ():
    texp = 0.05
    map = starry.Map(ydeg=2)
    np.random.seed(11)
    map[1:, :] = 0.1 * np.random.randn(8)
    orbit = exo.orbits.KeplerianOrbit(period=1.0, m_star=1.0, r_star=1.0)
    t = np.linspace(-0.2, 0.2, 10000)
    flux = map.flux(t=t, orbit=orbit, ro=0.1).eval()
    xo = orbit.get_relative_position(t)[0].eval()
    yo = orbit.get_relative_position(t)[1].eval()
    flux = map.flux(xo=xo, yo=yo, ro=0.1)
    fluence_mavg = moving_average(flux, int(texp / (t[1] - t[0])))
    fluence_starry = map.flux(t=t, orbit=orbit, ro=0.1, 
                              texp=texp, oversample=30).eval()
    fluence_starry_vec = map.flux(t=t, orbit=orbit, ro=0.1, 
                              texp=np.ones_like(t) * texp, oversample=30).eval()
    assert np.allclose(fluence_mavg, fluence_starry, fluence_starry_vec)


def test_ylm_phase():
    texp = 0.05
    map = starry.Map(ydeg=2)
    np.random.seed(11)
    map[1:, :] = 0.1 * np.random.randn(8)
    theta = np.linspace(0, 360, 10000)
    t = np.linspace(-0.2, 0.2, 10000)

    flux = map.flux(theta=theta)
    fluence_mavg = moving_average(flux, int(texp / (t[1] - t[0])))

    orbit = exo.orbits.KeplerianOrbit(period=1.0)
    fluence_starry = map.flux(t=t, orbit=orbit, theta=theta, texp=texp, ro=0.1, oversample=30).eval()

    import matplotlib.pyplot as plt
    plt.switch_backend("Qt5Agg")
    plt.plot(flux)
    plt.plot(fluence_starry)
    plt.plot(fluence_mavg)
    plt.show()

    assert np.allclose(fluence_mavg, fluence_starry)


if __name__ == "__main__":
    test_ylm_occ()
