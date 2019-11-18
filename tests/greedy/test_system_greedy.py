# -*- coding: utf-8 -*-
"""Test Keplerian system stuff."""
import starry
import pytest
import theano
import numpy as np


def test_orientation(Omega=45, inc=35):
    # Instantiate
    pri = starry.Primary(starry.Map(amp=0))
    sec = starry.Secondary(
        starry.Map(ydeg=1, amp=1),
        porb=1.0,
        r=0,
        m=0,
        inc=inc,
        Omega=Omega,
        prot=1.0,
        theta0=180.0,
    )
    sec.map[1, 0] = 1.0
    sys = starry.System(pri, sec)

    # Align the rotational axis with the orbital axis
    sec.map.inc = sec.inc
    sec.map.obl = sec.Omega

    # Compute the flux
    t = np.linspace(-0.5, 0.5, 1000)
    flux = sys.flux(t)

    # This is the analytic result
    flux_analytic = 1.0 - np.sin(inc * np.pi / 180.0) * (
        2.0 / np.sqrt(3.0)
    ) * np.cos(2 * np.pi * t)

    assert np.allclose(flux, flux_analytic)


def test_bodies():
    pri = starry.Primary(starry.Map())
    sec = starry.Secondary(starry.Map(ydeg=1), porb=1.0)
    sys = starry.System(pri, sec)
    assert sys.primary == pri
    assert sys.secondaries[0] == sec
