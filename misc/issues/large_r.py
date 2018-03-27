"""Test various numerical issues."""
import numpy as np
import matplotlib.pyplot as pl
import starry


def Earth():
    """
    For Earth in secondary eclipse behind the Sun, the instability
    begins at l ~ 4.
    """
    # Instantiate a system
    lmax = 4
    planet = starry.Planet(lmax, prot=0)
    star = starry.Star()
    system = starry.System([star, planet])

    # Zoom in on secondary eclipse ingress
    time = np.linspace(0.7114, 0.7122, 10000)
    fig, ax = pl.subplots(1, figsize=(8, 7))
    for m in range(-lmax, lmax + 1):
        if m == 3 or m == 1:
            # These I haven't fixed yet
            alpha = 0.25
        else:
            alpha = 1
        planet.map.reset()
        planet.map[0, 0] = 1
        planet.map[lmax, m] = 1
        system.compute(time)
        ax.plot(time, planet.flux, label="m = %d" % m, alpha=alpha)
    pl.legend(ncol=9, fontsize=6)
    pl.title("Secondary eclipse ingress for l = %d" % lmax)
    pl.show()


def EarthManual():
    """
    Emulate `Earth()` without the orbital module. Same thing!
    """
    lmax = 4
    yo = 0
    ro = 6.957e8 / 6.3781e6
    time = np.linspace(0, 1, 10000)
    xo = np.linspace((ro + 1) + 0.2, (ro - 1) - 0.2, 10000)
    ylm = starry.Map(lmax)
    fig, ax = pl.subplots(1, figsize=(8, 7))
    for m in range(-lmax, lmax + 1):
        if m == 3 or m == 1:
            # These I haven't fixed yet
            alpha = 0.25
        else:
            alpha = 1
        ylm.reset()
        ylm[0, 0] = 1
        ylm[lmax, m] = 1
        flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
        ax.plot(time, flux, label="m = %d" % m, alpha=alpha)
    pl.legend(ncol=9, fontsize=6)
    pl.title("Secondary eclipse ingress for l = %d" % lmax)


Earth()
