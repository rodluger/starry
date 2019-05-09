# -*- coding: utf-8 -*-
import exoplanet as exo
import starry
import numpy as np


def test_ld():
    map = starry.Map(udeg=2)
    map[1:] = [0.4, 0.26]
    orbit = exo.orbits.KeplerianOrbit(period=1.0, m_star=1.0, r_star=1.0)
    t = np.linspace(-0.25, 0.25, 100)

    # Compute the whole light curve with Theano
    f1 = map.flux(t=t, orbit=orbit, ro=0.1, use_in_transit=False).eval()

    # Compute just the transit with Theano
    f2 = map.flux(t=t, orbit=orbit, ro=0.1, use_in_transit=True).eval()

    # Compute the whole light curve without Theano
    coords = orbit.get_relative_position(t)
    xo = (coords[0] / orbit.r_star).eval()
    yo = (coords[1] / orbit.r_star).eval()
    b = np.sqrt(xo * xo + yo * yo)
    zo = -(coords[2] / orbit.r_star).eval()
    f3 = map.flux(b=b, zo=zo, ro=0.1)

    # They should all be the same!
    assert np.allclose(f1, f2, f3)


def test_ylm():
    map = starry.Map(ydeg=1)
    map[1, :] = [0.3, 0.2, 0.1]
    orbit = exo.orbits.KeplerianOrbit(period=1.0, m_star=1.0, r_star=1.0)
    t = np.linspace(-0.25, 0.25, 100)
    theta = 30.

    # Compute the whole light curve with Theano
    f1 = map.flux(t=t, orbit=orbit, ro=0.1, theta=theta, use_in_transit=False).eval()

    # Compute just the transit with Theano
    f2 = map.flux(t=t, orbit=orbit, ro=0.1, theta=theta, use_in_transit=True).eval()

    # Compute the whole light curve without Theano
    coords = orbit.get_relative_position(t)
    xo = (coords[0] / orbit.r_star).eval()
    yo = (coords[1] / orbit.r_star).eval()
    zo = -(coords[2] / orbit.r_star).eval()
    f3 = map.flux(xo=xo, yo=yo, zo=zo, ro=0.1, theta=theta)

    # They should all be the same!
    assert np.allclose(f1, f2, f3)


if __name__ == "__main__":
    test_ylm()
