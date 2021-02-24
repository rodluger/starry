# -*- coding: utf-8 -*-
"""Test Keplerian system stuff."""
import starry
import pytest
import theano
import numpy as np
import astropy.units as u
import exoplanet


def test_compare_to_map_rv():
    """Ensure we get the same result by calling `map.rv()` and `sys.rv()`.
    """
    # Define the map
    map = starry.Map(ydeg=1, udeg=2, rv=True, amp=1, veq=1, alpha=0)
    map[1, 0] = 0.5

    # Define the star
    A = starry.Primary(
        map, r=1.0, m=1.0, prot=0, length_unit=u.Rsun, mass_unit=u.Msun
    )

    # Define the planet
    b = starry.Secondary(
        starry.Map(rv=True, amp=1, veq=0),
        r=0.1,
        porb=1.0,
        m=0.01,
        t0=0.0,
        inc=86.0,
        ecc=0.3,
        w=60,
        length_unit=u.Rsun,
        mass_unit=u.Msun,
        angle_unit=u.degree,
        time_unit=u.day,
    )

    # Define the system
    sys = starry.System(A, b)

    # Time array
    time = np.linspace(-0.05, 0.05, 1000)

    # Get the positions of both bodies
    x, y, z = sys.position(time)

    # Compute the relative positions
    xo = x[1] - x[0]
    yo = y[1] - y[0]
    zo = z[1] - z[0]
    ro = b.r / A.r

    # Compare
    rv1 = map.rv(xo=xo, yo=yo, ro=ro)
    rv2 = sys.rv(time, keplerian=False)
    assert np.allclose(rv1, rv2)


def test_compare_to_exoplanet():
    """Ensure we get the same result with `starry` and `exoplanet`.
    """
    # Define the star
    A = starry.Primary(
        starry.Map(rv=True, veq=0),
        r=1.0,
        m=1.0,
        prot=0,
        length_unit=u.Rsun,
        mass_unit=u.Msun,
    )

    # Define the planet
    b = starry.Secondary(
        starry.Map(rv=True, veq=0),
        r=0.1,
        porb=1.0,
        m=0.01,
        t0=0.0,
        inc=86.0,
        ecc=0.3,
        w=60,
        length_unit=u.Rsun,
        mass_unit=u.Msun,
        angle_unit=u.degree,
        time_unit=u.day,
    )

    # Define the planet
    c = starry.Secondary(
        starry.Map(rv=True, veq=0),
        r=0.1,
        porb=1.7,
        m=0.02,
        t0=0.3,
        inc=87.0,
        ecc=0.2,
        w=70,
        length_unit=u.Rsun,
        mass_unit=u.Msun,
        angle_unit=u.degree,
        time_unit=u.day,
    )

    # Define the system
    sys = starry.System(A, b, c)

    # Compute with starry
    time = np.linspace(-0.5, 0.5, 1000)
    rv1 = sys.rv(time, keplerian=True, total=True)

    # Compute with exoplanet
    orbit = exoplanet.orbits.KeplerianOrbit(
        period=[1.0, 1.7],
        t0=[0.0, 0.3],
        incl=[86.0 * np.pi / 180, 87.0 * np.pi / 180],
        ecc=[0.3, 0.2],
        omega=[60 * np.pi / 180, 70 * np.pi / 180],
        m_planet=[0.01, 0.02],
        m_star=1.0,
        r_star=1.0,
    )
    rv2 = orbit.get_radial_velocity(time).eval().sum(axis=1)

    assert np.allclose(rv1, rv2)
