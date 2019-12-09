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


def test_integration():
    pri = starry.Primary(starry.Map(udeg=2), r=1.0)
    pri.map[1:] = [0.5, 0.25]
    sec = starry.Secondary(starry.Map(ydeg=1), porb=1.0, r=0.25)

    # Manual integration
    t = np.linspace(-0.1, 0.1, 10000)
    sys = starry.System(pri, sec, texp=0)
    flux = sys.flux(t)
    t = t.reshape(-1, 1000).mean(axis=1)
    flux = flux.reshape(-1, 1000).mean(axis=1)

    sys0 = starry.System(pri, sec, texp=0.02, order=0, oversample=999)
    assert sys0.order == 0
    assert sys0.oversample == 999
    flux0 = sys0.flux(t)
    assert np.allclose(flux, flux0)

    sys1 = starry.System(pri, sec, texp=0.02, order=1, oversample=999)
    assert sys1.order == 1
    assert sys1.oversample == 999
    flux1 = sys1.flux(t)
    assert np.allclose(flux, flux1)

    sys2 = starry.System(pri, sec, texp=0.02, order=2, oversample=999)
    assert sys2.order == 2
    assert sys2.oversample == 999
    flux2 = sys2.flux(t)
    assert np.allclose(flux, flux2)


def test_reflected_light():
    pri = starry.Primary(starry.Map(amp=0), r=1)
    sec = starry.Secondary(starry.Map(reflected=True), porb=1.0, r=1)
    sys = starry.System(pri, sec)
    t = np.concatenate((np.linspace(0.1, 0.4, 50), np.linspace(0.6, 0.9, 50)))
    flux = sys.flux(t)

    # TODO: Add an analytic validation here
