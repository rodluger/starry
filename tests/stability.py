"""Stability calculations in the b-r plane."""
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import starry
import numpy as np
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')
MACHINE_PRECISION = 1.6e-16
colors = pl.cm.coolwarm(np.linspace(0., 1., 256))
errormap = mcolors.LinearSegmentedColormap.from_list('errormap', colors)


def BRGrid(ax, l, m, blen=100, rlen=300, rmin=1e-5, rmax=1e3,
           bmin=1e-5, bmax=1e3, blog=True, bmid=None, ticklabels=[]):
    """Plot the error on the b-r plane."""
    r = np.logspace(np.log10(rmin), np.log10(rmax), rlen)

    # Exact
    m128 = starry.Map(l)
    m128.use_mp = True
    m128[l, m] = 1

    # Starry
    mt = starry.Map(l)
    mt[l, m] = 1

    # Compute
    flux_128 = np.zeros((blen, rlen))
    flux_starry = np.zeros((blen, rlen))
    for i, ro in enumerate(r):
        if blog:
            b = np.logspace(np.log10(bmin), np.log10(bmax), blen)
        else:
            b = np.linspace(bmin(ro), bmax(ro), blen)
        # Ensure the unstable value is in the array
        if bmid is not None:
            b[len(b) // 2] = bmid(ro)
        flux_128[:, i] = np.array(m128.flux(xo=0, yo=b, ro=ro))
        flux_starry[:, i] = np.array(mt.flux(xo=0, yo=b, ro=ro))

    # Compute the fractional error
    err_frac = (flux_starry - flux_128) / (flux_128)
    err_frac[(flux_128 == flux_starry)] = MACHINE_PRECISION
    err_frac = np.log10(np.abs(err_frac))

    # Compute the relative error
    err_rel = (flux_starry - flux_128)
    err_rel[(flux_128 == flux_starry)] = MACHINE_PRECISION
    err_rel = np.log10(np.abs(err_rel))

    # Plot
    if blog:
        extent = (np.log10(rmin), np.log10(rmax),
                  np.log10(bmin), np.log10(bmax))
    else:
        extent = (np.log10(rmin), np.log10(rmax), 0, 1)
    im1 = ax[0].imshow(err_frac, origin='lower', interpolation='none',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect='auto', cmap=errormap, extent=extent)
    im2 = ax[1].imshow(err_rel, origin='lower', interpolation='none',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect='auto', cmap=errormap, extent=extent)

    # Appearance
    for axis in ax:
        axis.set_xlabel(r'$\log\,r$', fontsize=10, labelpad=0)
        axis.set_xlim(np.log10(rmin), np.log10(rmax))
        axis.set_ylabel(r'$b$', fontsize=10)
        axis.get_yaxis().set_label_coords(-0.175, 0.5)
        if blog:
            axis.set_ylabel(r'$\log\,b$', fontsize=10)
        else:
            axis.set_ylim(0, 1)
            axis.set_ylabel(r'$b$', fontsize=10)
            axis.set_yticks([0, 0.5, 1])
            axis.set_yticklabels(ticklabels)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontsize(6)

    return im1, im2


def Ylm(l, m):
    """Generate a stability plot for a given Ylm."""
    fig = pl.figure(figsize=(8.5, 11))
    fig.subplots_adjust(top=0.925, bottom=0.075)
    ax1 = [pl.subplot2grid([108, 2], [0, 0], rowspan=30),
           pl.subplot2grid([108, 2], [0, 1], rowspan=30)]
    ax2 = [pl.subplot2grid([108, 2], [35, 0], rowspan=10),
           pl.subplot2grid([108, 2], [35, 1], rowspan=10)]
    ax3 = [pl.subplot2grid([108, 2], [46, 0], rowspan=10),
           pl.subplot2grid([108, 2], [46, 1], rowspan=10)]
    ax4 = [pl.subplot2grid([108, 2], [61, 0], rowspan=10),
           pl.subplot2grid([108, 2], [61, 1], rowspan=10)]
    ax5 = [pl.subplot2grid([108, 2], [72, 0], rowspan=10),
           pl.subplot2grid([108, 2], [72, 1], rowspan=10)]
    ax6 = [pl.subplot2grid([108, 2], [87, 0], rowspan=10),
           pl.subplot2grid([108, 2], [87, 1], rowspan=10)]
    ax7 = [pl.subplot2grid([108, 2], [98, 0], rowspan=10),
           pl.subplot2grid([108, 2], [98, 1], rowspan=10)]

    ax1[0].set_title('Fractional Error')
    ax1[1].set_title('Relative Error')

    Delta = 1e-1
    eps = 1e-6
    im, _ = BRGrid(ax1, l, m)
    BRGrid(ax2, l, m, bmin=lambda ro: ro - 1, bmax=lambda ro: ro + 1,
           blog=False, bmid=lambda ro: ro,
           ticklabels=[r"$r - 1$", r"$r$", r"$r + 1$"])
    BRGrid(ax3, l, m, bmin=lambda ro: ro - eps, bmax=lambda ro: ro + eps,
           blog=False, bmid=lambda ro: ro,
           ticklabels=[r"$r - \epsilon$", r"$r$", r"$r + \epsilon$"])
    BRGrid(ax4, l, m, bmin=lambda ro: ro + 1 - Delta,
           bmax=lambda ro: ro + 1 + Delta,
           blog=False, bmid=lambda ro: ro + 1,
           ticklabels=[r"$r + 1 - \Delta$", r"$r + 1$", r"$r + 1 + \Delta$"])
    BRGrid(ax5, l, m, bmin=lambda ro: ro + 1 - eps,
           bmax=lambda ro: ro + 1 + eps,
           blog=False, bmid=lambda ro: ro + 1,
           ticklabels=[r"$r + 1 - \epsilon$", r"$r + 1$",
                       r"$r + 1 + \epsilon$"])
    BRGrid(ax6, l, m, bmin=lambda ro: ro - 1 - Delta,
           bmax=lambda ro: ro - 1 + Delta,
           blog=False, bmid=lambda ro: ro - 1,
           ticklabels=[r"$r - 1 - \Delta$", r"$r - 1$", r"$r - 1 + \Delta$"])
    BRGrid(ax7, l, m, bmin=lambda ro: ro - 1 - eps,
           bmax=lambda ro: ro - 1 + eps,
           blog=False, bmid=lambda ro: ro - 1,
           ticklabels=[r"$r - 1 - \epsilon$", r"$r - 1$",
                       r"$r - 1 + \epsilon$"])

    # Labels for zoomed-in regions
    pl.figtext(0.92, 0.565, r"$b = r$", ha="center", va="center", rotation=90,
               fontsize=28)
    pl.figtext(0.92, 0.365, r"$b = r + 1$", ha="center", va="center",
               rotation=90, fontsize=24)
    pl.figtext(0.92, 0.16, r"$b = r - 1$", ha="center", va="center",
               rotation=90, fontsize=24)

    # Hack a colorbar
    for axis in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        for ax in axis:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.axis('off')
    divider = make_axes_locatable(ax1[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'$\log\,\mathrm{error}$', fontsize=10)

    pl.suptitle(r"$Y_{%d,%d}$" % (l, m), fontsize=24, y=0.98)
    fig.savefig("Y_%d,%d.pdf" % (l, m))
    pl.close()


def Run(lmax=4):
    """Run the calculations for all Ylms."""
    for l in range(lmax + 1):
        print("Running l = %d..." % l)
        for m in tqdm(range(-l, l + 1)):
            Ylm(l, m)


if __name__ == "__main__":
    Run()
