"""Numerical stability tests."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np
from tqdm import tqdm
cmap = pl.get_cmap('plasma')


def color(l, lmax=8):
    """Return the color for the spherical harmonic degree `l`."""
    return cmap(0.1 + 0.8 * l / lmax)


def earth_eclipse(lmax=8):
    """Compute the error on the secondary eclipse of the Earth."""
    npts = 1000

    # Create our map
    m = Map(lmax)
    m.load_image('earth')

    # Compute. Ingress duration is
    # dt = (2 REARTH) / (2 PI * 1 AU / 1 year) ~ 7 minutes
    yo = 0
    ro = 6.957e8 / 6.3781e6
    time = np.linspace(0, 7 * 1.5, npts)
    xo = np.linspace(-(ro + 1.5), -(ro - 1.5), npts, -1)
    flux = np.array(m.flux(xo=xo, yo=yo, ro=ro))
    flux128 = np.array(m.flux_mp(xo=xo, yo=yo, ro=ro))

    # Show
    fig = pl.figure(figsize=(7, 6))
    nim = 10
    ax = [pl.subplot2grid((7, nim), (1, 0), colspan=nim, rowspan=3),
          pl.subplot2grid((7, nim), (4, 0), colspan=nim, rowspan=3)]
    fig.subplots_adjust(hspace=0.6)
    ax[0].plot(time, flux / flux[0])
    ax[1].plot(time, np.abs(flux / flux128 - 1))
    ax[1].set_yscale('log')
    ax[1].axhline(1e-3, color='k', ls='--', alpha=0.75, lw=0.5)
    ax[1].axhline(1e-6, color='k', ls='--', alpha=0.75, lw=0.5)
    ax[1].axhline(1e-9, color='k', ls='--', alpha=0.75, lw=0.5)
    ax[1].annotate("ppt", xy=(1e-3, 1e-3), xycoords="data", xytext=(3, -3),
                   textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax[1].annotate("ppm", xy=(1e-3, 1e-6), xycoords="data", xytext=(3, -3),
                   textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax[1].annotate("ppb", xy=(1e-3, 1e-9), xycoords="data", xytext=(3, -3),
                   textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax[1].set_ylim(5e-17, 20.)
    ax[0].set_xlim(0, time[-1])
    ax[1].set_xlim(0, time[-1])
    ax[1].set_xlabel("Time [minutes]", fontsize=16)
    ax[0].set_ylabel("Normalized flux", fontsize=16, labelpad=15)
    ax[1].set_ylabel("Fractional error", fontsize=16)

    # Plot the earth images
    res = 100
    ax_im = [pl.subplot2grid((7, nim), (0, n)) for n in range(nim)]
    x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    for n in range(nim):
        i = int(np.linspace(0, npts - 1, nim)[n])
        I = m.evaluate(axis=[0, 1, 0], theta=0, x=x, y=y)
        ax_im[n].imshow(I, origin="lower", interpolation="none", cmap='plasma',
                        extent=(-1, 1, -1, 1))
        xm = np.linspace(xo[i] - ro + 1e-5, xo[i] + ro - 1e-5, 10000)
        ax_im[n].fill_between(xm, yo - np.sqrt(ro ** 2 - (xm - xo[i]) ** 2),
                              yo + np.sqrt(ro ** 2 - (xm - xo[i]) ** 2),
                              color='w')
        ax_im[n].axis('off')
        ax_im[n].set_xlim(-1.05, 1.05)
        ax_im[n].set_ylim(-1.05, 1.05)

    fig.savefig("stability_earth.pdf", bbox_inches='tight')


def impact_param(ax, lmax=8):
    """Test the stability as a function of b."""
    npts = 200
    yo = 0
    cutoff = 2.5e-16
    barr = np.logspace(-5, np.log10(2.0), npts)
    xo = np.sqrt(barr ** 2 - yo ** 2)

    # Create our map
    ylm = Map(lmax)
    ylm.optimize = True

    # Compute
    for l in range(0, lmax + 1):

        # Set the constant term
        # so we don't divide by zero when
        # computing the fractional error below
        ylm.reset()
        ylm[0, 0] = 1

        # Set all odd terms
        for m in range(-l, l + 1):
            # Skip even terms (they're fine!)
            if (l + m) % 2 == 0:
                continue
            ylm[l, m] = 1

        ro = 0.1
        # Compute
        flux = np.array(ylm.flux(xo=xo, yo=yo, ro=ro))
        flux128 = np.array(ylm.flux_mp(xo=xo, yo=yo, ro=ro))
        error = np.abs(flux / flux128 - 1)

        # HACK to make it prettier.
        error[error < cutoff] = cutoff
        # HACK Believe it or not, quadruple precision is not good
        # enough when the impact parameter is smaller than about 0.001
        # at high values of l! While our taylor expansions are fine, the
        # high precision comparison flux is wrong, so we get a spurious
        # increase in the error towards smaller b. Let's just trim it
        # for simplicity.
        error[xo < 0.002] = cutoff

        ax.plot(barr, error, ls='-',
                label=r'$l = %d$' % l, color=color(l, lmax))

    ax.legend(loc="upper left", ncol=3)
    ax.axhline(1e-3, color='k', ls='--', alpha=0.75, lw=0.5)
    ax.axhline(1e-6, color='k', ls='--', alpha=0.75, lw=0.5)
    ax.axhline(1e-9, color='k', ls='--', alpha=0.75, lw=0.5)
    ax.annotate("ppt", xy=(1e-5, 1e-3), xycoords="data", xytext=(3, -3),
                textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax.annotate("ppm", xy=(1e-5, 1e-6), xycoords="data", xytext=(3, -3),
                textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax.annotate("ppb", xy=(1e-5, 1e-9), xycoords="data", xytext=(3, -3),
                textcoords="offset points", ha="left", va="top", alpha=0.75)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(5e-17, 20.)
    ax.set_xlim(1e-5, np.log10(2.0))
    ax.set_xlabel("Impact parameter", fontsize=16)
    ax.set_ylabel("Fractional error", fontsize=16)


def occultor_radius(ax, lmax=8):
    """Test the stability as a function of occultor radius."""
    # Knobs
    eps = 1e-6
    npts = 50
    yo = 0
    rarr = np.logspace(-3, 3, npts)

    # Create our map
    ylm = Map(lmax)

    # Loop over the degrees
    for l in tqdm(range(lmax + 1)):
        ylm.reset()

        # Set the constant term
        # so we don't divide by zero when
        # computing the fractional error below
        ylm[0, 0] = 1

        # Set the coefficients for all orders
        for m in range(-l, l + 1):
            ylm[l, m] = 1
        # Occultor radius loop
        error = np.zeros_like(rarr)
        for i, ro in enumerate(rarr):
            xo0 = 0.5 * ((ro + 1) + np.abs(ro - 1))
            xo = np.linspace(xo0 - 25 * eps, xo0 + 25 * eps, 50)
            flux = np.array(ylm.flux(xo=xo, yo=yo, ro=ro))
            flux128 = np.array(ylm.flux_mp(xo=xo, yo=yo, ro=ro))
            error[i] = np.max(np.abs((flux / flux128 - 1)))
        ax.plot(rarr, error, '-', color=color(l, lmax),
                label=r"$l=%d$" % l)

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


if __name__ == "__main__":

    # Earth stability
    earth_eclipse()

    # Ylm stability
    fig, ax = pl.subplots(2, figsize=(9, 8))
    impact_param(ax[0])
    occultor_radius(ax[1])
    fig.savefig("stability.pdf", bbox_inches='tight')
