# -*- coding: utf-8 -*-
"""
Functions to compare stuff to the beta version.

"""
import starry
import starry_beta
import numpy as np
import pytest
from astropy import constants, units

np.random.seed(12)
G_grav = constants.G.to(units.R_sun ** 3 / units.M_sun / units.day ** 2).value


def test_edge_on_eccentric():
    # Params
    ydeg = 10
    u = [0.5, 0.25]
    y = 0.1 * np.random.randn((ydeg + 1) ** 2 - 1)
    porb = 1.0
    prot = 1.0
    amp = 0.25
    r = 0.5
    m = 0.25
    ecc = 0.5
    w = 75
    t = np.linspace(-0.75, 0.75, 10000)

    # Beta version
    pri_beta = starry_beta.kepler.Primary(lmax=2)
    pri_beta[1], pri_beta[2] = u
    sec_beta = starry_beta.kepler.Secondary(lmax=ydeg)
    sec_beta[1:, :] = y
    sec_beta.porb = porb
    sec_beta.prot = prot
    sec_beta.L = amp
    sec_beta.r = r
    sec_beta.a = (G_grav * (1.0 + m) * porb ** 2 / (4 * np.pi ** 2)) ** (
        1.0 / 3
    )
    sec_beta.inc = 90
    sec_beta.Omega = 0
    sec_beta.ecc = ecc
    sec_beta.w = w
    sys_beta = starry_beta.kepler.System(pri_beta, sec_beta)
    sys_beta.compute(t)
    flux_beta = np.array(sys_beta.lightcurve)

    # Compute the time of transit
    M0 = 0.5 * np.pi - w * np.pi / 180.0
    f = M0
    E = np.arctan2(np.sqrt(1 - ecc ** 2) * np.sin(f), ecc + np.cos(f))
    M = E - ecc * np.sin(E)
    t0 = (M - M0) * porb / (2 * np.pi)

    # Compute the time of eclipse
    E = np.arctan2(
        np.sqrt(1 - ecc ** 2) * np.sin(f + np.pi), ecc + np.cos(f + np.pi)
    )
    M = E - ecc * np.sin(E)
    t_ecl = (M - M0) * porb / (2 * np.pi)

    # This is the required phase offset such that the map coefficients
    # correspond to what the observer sees at secondary eclipse
    theta0 = -(t_ecl - t0) * 360

    # Version 1
    pri = starry.Primary(starry.Map(udeg=2))
    pri.map[1:] = u
    sec = starry.Secondary(
        starry.Map(ydeg=ydeg, amp=amp),
        porb=porb,
        r=r,
        m=m,
        inc=90,
        Omega=0,
        ecc=ecc,
        w=w,
        t0=t0,
        theta0=theta0,
    )
    sec.map[1:, :] = y
    sys = starry.System(pri, sec)
    flux = sys.flux(t)

    # Compare
    assert np.allclose(flux, flux_beta)
