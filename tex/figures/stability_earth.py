"""Numerical stability tests."""
import starry2
import matplotlib.pyplot as pl
import numpy as np
cmap = pl.get_cmap('plasma')


def earth_eclipse(lmax=20):
    """Compute the error on the secondary eclipse of the Earth."""
    npts = 1000

    # Create our map
    map = starry2.Map(lmax)
    map.load_image('earth')

    # Compute. Ingress duration is
    # dt = (2 REARTH) / (2 PI * 1 AU / 1 year) ~ 7 minutes
    yo = 0
    ro = 6.957e8 / 6.3781e6
    time = np.linspace(0, 7 * 1.5, npts)
    xo = np.linspace(-(ro + 1.5), -(ro - 1.5), npts, -1)
    flux = np.array(map.flux(xo=xo, yo=yo, ro=ro))

    # Compute at high precision
    map_128 = starry2.Map(lmax, multi=True)
    map_128[:, :] = map[:, :]
    flux128 = np.array(map_128.flux(xo=xo, yo=yo, ro=ro))

    # Show
    fig = pl.figure(figsize=(7, 6))
    nim = 10
    ax = [pl.subplot2grid((7, nim), (1, 0), colspan=nim, rowspan=3),
          pl.subplot2grid((7, nim), (4, 0), colspan=nim, rowspan=3)]
    fig.subplots_adjust(hspace=0.6)
    ax[0].plot(time, flux / flux[0])
    ax[1].plot(time, np.abs(flux - flux128))
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
    ax[1].set_ylabel("Relative error", fontsize=16)

    # Plot the earth images
    res = 100
    ax_im = [pl.subplot2grid((7, nim), (0, n)) for n in range(nim)]
    x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    map.axis = [0, 1, 0]
    for n in range(nim):
        i = int(np.linspace(0, npts - 1, nim)[n])
        I = [map(theta=0, x=x[j], y=y[j]) for j in range(res)]
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


if __name__ == "__main__":

    # Earth stability
    earth_eclipse()
