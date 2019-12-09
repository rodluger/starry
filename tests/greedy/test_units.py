# -*- coding: utf-8 -*-
"""
Unit unit tests.

"""
import starry
import astropy.units as u
from astropy.constants import G
import numpy as np
import pytest


def test_default_body_units():
    body = starry.Primary(starry.Map())
    assert body.length_unit == u.Rsun
    assert body.mass_unit == u.Msun
    assert body.angle_unit == u.degree
    assert body.time_unit == u.day
    assert body.map.angle_unit == u.degree


def test_default_system_units():
    pri = starry.Primary(starry.Map())
    sec = starry.Secondary(starry.Map(), porb=1.0)
    sys = starry.System(pri, sec)
    assert sys.time_unit == u.day


def test_unit_conversions():
    body = starry.Secondary(
        starry.Map(ydeg=1, inc=1.0, obl=1.0, angle_unit=u.radian),
        r=1.0,
        m=1.0,
        inc=1.0,
        porb=1.0,
        prot=1.0,
        t0=1.0,
        theta0=1.0,
        Omega=1.0,
        length_unit=u.cm,
        mass_unit=u.gram,
        time_unit=u.second,
        angle_unit=u.radian,
    )
    assert body.r == 1.0
    assert body._r == body.length_unit.in_units(u.Rsun)
    assert body.m == 1.0
    assert body._m == body.mass_unit.in_units(u.Msun)
    assert body.inc == 1.0
    assert body._inc == body.angle_unit.in_units(u.radian)
    assert body.theta0 == 1.0
    assert body._theta0 == body.angle_unit.in_units(u.radian)
    assert body.Omega == 1.0
    assert body._Omega == body.angle_unit.in_units(u.radian)
    assert body.prot == 1.0
    assert body._prot == body.time_unit.in_units(u.day)
    assert body.t0 == 1.0
    assert body._t0 == body.time_unit.in_units(u.day)
    assert body.ecc == 0.0
    assert body.map.inc == 1.0
    assert body.map._inc == body.map.angle_unit.in_units(u.radian)
    assert body.map.obl == 1.0
    assert body.map._obl == body.map.angle_unit.in_units(u.radian)

    # First test the orbital period
    assert body.porb == 1.0
    assert body._porb == body.time_unit.in_units(u.day)

    # Now test the semi-major axis
    assert body.a is None
    body.a = 1.0
    assert body.a == 1.0
    assert body._a == body.length_unit.in_units(u.Rsun)
    assert body.porb is None


def test_omega_w():
    """Ensure `w` and `omega` are equivalent."""
    body = starry.Secondary(starry.Map(), porb=1.0, w=30.0)
    assert np.allclose(body.omega, 30)
    body.omega = 60
    assert np.allclose(body.w, 60)

    body = starry.Secondary(starry.Map(), porb=1.0, omega=30.0)
    assert np.allclose(body.w, 30)
    body.w = 60
    assert np.allclose(body.omega, 60)


def test_period_semi():
    # Check that an error is raised if neither a nor porb is given
    with pytest.raises(ValueError) as e:
        body = starry.Secondary(starry.Map())
    assert "Must provide a value for either `porb` or `a`" in str(e.value)

    # Check that the semi --> period conversion works
    pri = starry.Primary(starry.Map(), m=1.0, mass_unit=u.Msun)
    sec = starry.Secondary(
        starry.Map(), a=10.0, m=1.0, length_unit=u.AU, mass_unit=u.Mearth
    )
    sys = starry.System(pri, sec)
    period = sys._get_periods()[0]
    true_period = (
        (
            (2 * np.pi)
            * (sec.a * sec.length_unit) ** (3 / 2)
            / (np.sqrt(G * (pri.m * pri.mass_unit + sec.m * sec.mass_unit)))
        )
        .to(u.day)
        .value
    )
    assert np.allclose(period, true_period)
