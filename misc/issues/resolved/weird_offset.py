"""Investigate a weird offset."""
import numpy as np
import matplotlib.pyplot as pl
import starry


def WeirdOffset1():
    """
    There's a really small offset in the flux right before and after first
    contact for most of the Ylms. Not really noticeable until you get to
    4th order or so, and mostly for large occultors. But I can get it
    to happen for the Y_{0,0} harmonic, so this should be something obvious!
    """
    lmax = 0
    yo = 0
    time = np.linspace(0, 1, 100)
    ylm = starry.Map(lmax)
    fig, ax = pl.subplots(1, figsize=(8, 7))
    ro = 100
    ylm.reset()
    ylm[0, 0] = 1
    eps = 0.0000001
    xo = np.linspace((ro + 1) + eps, (ro + 1) - eps, 100)

    ylm.use_mp = False
    flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
    ax.plot(time, flux)

    ylm.use_mp = True
    flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
    ax.plot(time, flux)

    pl.title("Secondary eclipse ingress for l = %d" % lmax)
    pl.show()


def WeirdOffset2():
    """
    The offset only happens for even nu terms.
    """
    lmax = 2
    eps = 1e-5
    ro = 100
    yo = 0
    xo = np.linspace((ro + 1) + eps, (ro + 1) - eps, 100)
    time = np.linspace(0, 1, 100)
    ylm = starry.Map(lmax)
    ylm.use_mp = True
    fig, ax = pl.subplots(1, figsize=(8, 7))

    # Odd nu
    ylm.reset()
    ylm[2, 1] = 1
    flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
    ax.plot(time, flux, label='nu odd')

    # Even nu
    ylm.reset()
    ylm[2, 2] = 1
    flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
    ax.plot(time, flux, label='nu even')

    pl.axhline(9.48e-10, color='k', alpha=0.5, lw=1)
    pl.ylim(-6e-10, 1.1e-9)
    pl.legend()
    pl.title("Secondary eclipse ingress for l = %d" % lmax)
    pl.show()


WeirdOffset1()
