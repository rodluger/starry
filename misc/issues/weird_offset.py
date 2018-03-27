"""Investigate a weird offset."""
import numpy as np
import matplotlib.pyplot as pl
import starry


def WeirdOffset():
    """
    There's a really small offset in the flux right before and after first
    contact for most of the Ylms. Not really noticeable until you get to
    4th order or so, and mostly for large occultors. But I can get it
    to happen for the Y_{0,0} harmonic, so this should be something obvious!
    Increasing the MP digits and/or increasing the precision of the elliptic
    integrals doesn't seem to help. Increasing the number of digits in M_PI
    doesn't either.
    """
    lmax = 0
    yo = 0
    time = np.linspace(0, 1, 100)
    ylm = starry.Map(lmax)
    ylm.use_mp = True
    fig, ax = pl.subplots(1, figsize=(8, 7))
    ro = 100
    ylm.reset()
    ylm[0, 0] = 1
    xo = np.linspace((ro + 1) + 0.00000001, (ro + 1) - 0.00000001, 100)
    flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
    ax.plot(time, flux)
    pl.title("Secondary eclipse ingress for l = %d" % lmax)
    pl.show()


WeirdOffset()
