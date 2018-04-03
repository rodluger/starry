"""Numerical stability tests."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np
from tqdm import tqdm
cmap = pl.get_cmap('plasma')
lmax = 8


def color(l):
    """Return the color for the spherical harmonic degree `l`."""
    return cmap(0.1 + 0.8 * l / lmax)


def impact_param():
    """Test the stability as a function of b."""
    npts = 500
    yo = 0
    ro = 0.1
    barr = np.logspace(-5, np.log10(1 + ro), npts)
    xo = np.sqrt(barr ** 2 - yo ** 2)

    # Double precision
    ylm = Map(lmax)

    # Quad precision (~exact)
    ylm128 = Map(lmax)
    ylm128.use_mp = True

    # Set
    ylm[2, 1] = 1
    ylm128[2, 1] = 1

    # Compute
    flux = np.array(ylm.flux(xo=xo, yo=yo, ro=ro))
    flux128 = np.array(ylm128.flux(xo=xo, yo=yo, ro=ro))

    # debug
    pl.switch_backend('MacOSX')
    pl.plot(barr, np.abs(flux / flux128 - 1))
    pl.xscale('log')
    pl.yscale('log')
    pl.show()
    quit()


def occultor_radius():
    """Test the stability as a function of occultor radius."""
    # Knobs
    eps = 1e-6
    npts = 50
    yo = 0
    rarr = np.logspace(-3, 3, npts)

    # Double precision
    ylm = Map(lmax)

    # Quad precision (~exact)
    ylm128 = Map(lmax)
    ylm128.use_mp = True

    # Set up
    fig, ax = pl.subplots(1, figsize=(9, 4))

    # Loop over the degrees
    for l in tqdm(range(lmax + 1)):
        ylm.reset()
        ylm128.reset()
        # Set the coefficients for all orders
        for m in range(-l, l + 1):
            ylm[l, m] = 1
            ylm128[l, m] = 1
        # Occultor radius loop
        error = np.zeros_like(rarr)
        for i, ro in enumerate(rarr):
            xo0 = 0.5 * ((ro + 1) + np.abs(ro - 1))
            xo = np.linspace(xo0 - 25 * eps, xo0 + 25 * eps, 50)
            flux = np.array(ylm.flux(xo=xo, yo=yo, ro=ro))
            flux128 = np.array(ylm128.flux(xo=xo, yo=yo, ro=ro))
            error[i] = np.max(np.abs((flux / flux128 - 1)))
        ax.plot(rarr, error, '-', color=color(l), lw=1, label=r"$l=%d$" % l)

    ax.legend(loc="upper left", ncol=3)
    ax.axhline(1e-3, color='k', ls='--', alpha=0.75, lw=0.5)
    ax.axhline(1e-6, color='k', ls='--', alpha=0.75, lw=0.5)
    ax.axhline(1e-9, color='k', ls='--', alpha=0.75, lw=0.5)
    ax.annotate("ppt", xy=(1e-3, 1e-3), xycoords="data", xytext=(3, -3),
                textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax.annotate("ppm", xy=(1e-3, 1e-6), xycoords="data", xytext=(3, -3),
                textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax.annotate("ppb", xy=(1e-3, 1e-9), xycoords="data", xytext=(3, -3),
                textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(5e-17, 20.)
    ax.set_xlim(1e-3, 1e3)
    ax.set_xlabel("Occultor radius", fontsize=16)
    ax.set_ylabel("Fractional error", fontsize=16)
    fig.savefig("stability.pdf", bbox_inches='tight')


if __name__ == "__main__":
    #impact_param()
    occultor_radius()
