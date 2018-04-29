"""Stability in the b-r plane for linear limb darkening (Mandel & Agol)."""
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import starry
import numpy as np
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')
MACHINE_PRECISION = 1.6e-16
MIN_FLUX = 1.e-9
colors = pl.cm.coolwarm(np.linspace(0., 1., 256))
errormap = mcolors.LinearSegmentedColormap.from_list('errormap', colors)


def BRGrid(ax, l, m, res=1, aspect='auto'):
    """Plot b-r grid."""
    blen = int(301 * res)
    rlen = int(301 * res)
    bmin = 1e-5
    bmax = 1e3
    rmin = 1e-5
    rmax = 1e3
    b = np.logspace(np.log10(bmin), np.log10(bmax), blen)
    r = np.logspace(np.log10(rmin), np.log10(rmax), rlen)
    R, B = np.meshgrid(r, b)

    # Exact
    m128 = starry.Map(l)
    m128.use_mp = True
    m128[l, m] = 1

    # Starry
    mt = starry.Map(l)
    mt[l, m] = 1

    # Starry (no taylor)
    mnt = starry.Map(l)
    mnt.optimize = False
    mnt[l, m] = 1

    # Compute
    flux_128 = np.array(m128.flux(xo=0, yo=B, ro=R))
    flux_starry = np.array(mt.flux(xo=0, yo=B, ro=R))
    flux_notaylor = np.array(mnt.flux(xo=0, yo=B, ro=R))

    # Compute the fractional errors
    err_starry = (flux_starry - flux_128) / (flux_128)
    err_starry[(flux_128 == 0) & (flux_starry == 0)] = MACHINE_PRECISION
    err_starry[(flux_128 == flux_starry)] = MACHINE_PRECISION
    err_starry = np.log10(np.abs(err_starry))

    err_notaylor = (flux_notaylor - flux_128) / flux_128
    err_notaylor[(flux_128 == 0) & (flux_notaylor == 0)] = MACHINE_PRECISION
    err_notaylor[(flux_128 == flux_notaylor)] = MACHINE_PRECISION
    err_notaylor = np.log10(np.abs(err_notaylor))

    # Mask regions where the flux is very small
    err_starry_mask = np.zeros_like(err_starry)
    err_starry_mask[(np.abs(flux_128) < MIN_FLUX) &
                    (np.abs(flux_starry) < MIN_FLUX) &
                    (err_starry > np.log10(MACHINE_PRECISION))] = 1
    err_notaylor_mask = np.zeros_like(err_notaylor)
    err_notaylor_mask[(np.abs(flux_128) < MIN_FLUX) &
                      (np.abs(flux_notaylor) < MIN_FLUX) &
                      (err_notaylor > np.log10(MACHINE_PRECISION))] = 1

    # Plot
    im1 = ax[0].imshow(err_starry, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect=aspect, cmap=errormap,
                       extent=(np.log10(rmin), np.log10(rmax),
                               np.log10(bmin), np.log10(bmax)))

    im2 = ax[1].imshow(err_notaylor, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect=aspect, cmap=errormap,
                       extent=(np.log10(rmin), np.log10(rmax),
                               np.log10(bmin), np.log10(bmax)))

    # Masks
    cmap_mask = pl.get_cmap('Greys')
    cmap_mask.set_under(alpha=0)
    cmap_mask.set_over(errormap(0))
    ax[0].imshow(err_starry_mask, origin='lower',
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax),
                         np.log10(bmin), np.log10(bmax)), aspect=aspect)
    ax[1].imshow(err_notaylor_mask, origin='lower',
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax),
                         np.log10(bmin), np.log10(bmax)), aspect=aspect)

    # Appearance
    for axis in ax:
        axis.set_xlabel(r'$\log\,r$', fontsize=10, labelpad=0)
        axis.set_xlim(np.log10(rmin), np.log10(rmax))
        axis.set_ylim(np.log10(bmin), np.log10(bmax))
        axis.set_ylabel(r'$\log\,b$', fontsize=10)
        axis.get_yaxis().set_label_coords(-0.175, 0.5)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontsize(6)

    return im1, im2


def RMinusOneRPlusOne(ax, l, m, res=1, eps=1, aspect='auto'):
    """Plot the error for r - 1 < b < r + 1."""
    blen = int(101 * res)
    rlen = int(301 * res)
    rmin = 1e-5
    rmax = 1e3
    r = np.logspace(np.log10(rmin), np.log10(rmax), rlen)

    # Exact
    m128 = starry.Map(l)
    m128.use_mp = True
    m128[l, m] = 1

    # Starry
    mt = starry.Map(l)
    mt[l, m] = 1

    # Starry (no taylor)
    mnt = starry.Map(l)
    mnt.optimize = False
    mnt[l, m] = 1

    # Compute
    flux_128 = np.zeros((blen, rlen))
    flux_starry = np.zeros((blen, rlen))
    flux_notaylor = np.zeros((blen, rlen))
    for i, ro in enumerate(r):
        b = np.linspace(ro - eps, ro + eps, blen)
        # Ensure the unstable value is in the array
        b[len(b) // 2] = ro
        flux_128[:, i] = np.array(m128.flux(xo=0, yo=b, ro=ro))
        flux_starry[:, i] = np.array(mt.flux(xo=0, yo=b, ro=ro))
        flux_notaylor[:, i] = np.array(mnt.flux(xo=0, yo=b, ro=ro))

    # Compute the fractional errors
    err_starry = (flux_starry - flux_128) / (flux_128)
    err_starry[(flux_128 == 0) & (flux_starry == 0)] = MACHINE_PRECISION
    err_starry[(flux_128 == flux_starry)] = MACHINE_PRECISION
    err_starry = np.log10(np.abs(err_starry))

    err_notaylor = (flux_notaylor - flux_128) / flux_128
    err_notaylor[(flux_128 == 0) & (flux_notaylor == 0)] = MACHINE_PRECISION
    err_notaylor[(flux_128 == flux_notaylor)] = MACHINE_PRECISION
    err_notaylor = np.log10(np.abs(err_notaylor))

    # Mask regions where the flux is very small
    err_starry_mask = np.zeros_like(err_starry)
    err_starry_mask[(np.abs(flux_128) < MIN_FLUX) &
                    (np.abs(flux_starry) < MIN_FLUX) &
                    (err_starry > np.log10(MACHINE_PRECISION))] = 1
    err_notaylor_mask = np.zeros_like(err_notaylor)
    err_notaylor_mask[(np.abs(flux_128) < MIN_FLUX) &
                      (np.abs(flux_notaylor) < MIN_FLUX) &
                      (err_notaylor > np.log10(MACHINE_PRECISION))] = 1

    # Plot
    im1 = ax[0].imshow(err_starry, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect=aspect, cmap=errormap,
                       extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    im2 = ax[1].imshow(err_notaylor, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect=aspect, cmap=errormap,
                       extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    # Masks
    cmap_mask = pl.get_cmap('Greys')
    cmap_mask.set_under(alpha=0)
    cmap_mask.set_over(errormap(0))
    ax[0].imshow(err_starry_mask, origin='lower', aspect=aspect,
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax), 0, 2))
    ax[1].imshow(err_notaylor_mask, origin='lower', aspect=aspect,
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    # Appearance
    for axis in ax:
        axis.set_xlabel(r'$\log\,r$', fontsize=10, labelpad=0)
        axis.set_xlim(np.log10(rmin), np.log10(rmax))
        axis.set_ylim(0, 2)
        axis.set_ylabel(r'$b$', fontsize=10)
        axis.get_yaxis().set_label_coords(-0.175, 0.5)
        if eps == 1:
            axis.set_yticks([0, 1, 2])
            axis.set_yticklabels([r"$r - 1$", r"$r$", r"$r + 1$"])
        else:
            axis.set_yticks([0, 1, 2])
            axis.set_yticklabels([r"$r - \epsilon$", r"$r$",
                                  r"$r + \epsilon$"])
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontsize(6)

    return im1, im2


def RPlusOne(ax, l, m, res=1, eps=0.1, aspect='auto'):
    """Plot the error for b ~ r + 1."""
    blen = int(101 * res)
    rlen = int(301 * res)
    rmin = 1e-5
    rmax = 1e3
    r = np.logspace(np.log10(rmin), np.log10(rmax), rlen)

    # Exact
    m128 = starry.Map(l)
    m128.use_mp = True
    m128[l, m] = 1

    # Starry
    mt = starry.Map(l)
    mt[l, m] = 1

    # Starry (no taylor)
    mnt = starry.Map(l)
    mnt.optimize = False
    mnt[l, m] = 1

    # Compute
    flux_128 = np.zeros((blen, rlen))
    flux_starry = np.zeros((blen, rlen))
    flux_notaylor = np.zeros((blen, rlen))
    for i, ro in enumerate(r):
        b = np.linspace(ro + 1 - eps, ro + 1 + eps, blen)
        # Ensure the unstable value is in the array
        b[len(b) // 2] = ro + 1
        flux_128[:, i] = np.array(m128.flux(xo=0, yo=b, ro=ro))
        flux_starry[:, i] = np.array(mt.flux(xo=0, yo=b, ro=ro))
        flux_notaylor[:, i] = np.array(mnt.flux(xo=0, yo=b, ro=ro))

    # Compute the fractional errors
    err_starry = (flux_starry - flux_128) / (flux_128)
    err_starry[(flux_128 == 0) & (flux_starry == 0)] = MACHINE_PRECISION
    err_starry[(flux_128 == flux_starry)] = MACHINE_PRECISION
    err_starry = np.log10(np.abs(err_starry))

    err_notaylor = (flux_notaylor - flux_128) / flux_128
    err_notaylor[(flux_128 == 0) & (flux_notaylor == 0)] = MACHINE_PRECISION
    err_notaylor[(flux_128 == flux_notaylor)] = MACHINE_PRECISION
    err_notaylor = np.log10(np.abs(err_notaylor))

    # Mask regions where the flux is very small
    err_starry_mask = np.zeros_like(err_starry)
    err_starry_mask[(np.abs(flux_128) < MIN_FLUX) &
                    (np.abs(flux_starry) < MIN_FLUX) &
                    (err_starry > np.log10(MACHINE_PRECISION))] = 1
    err_notaylor_mask = np.zeros_like(err_notaylor)
    err_notaylor_mask[(np.abs(flux_128) < MIN_FLUX) &
                      (np.abs(flux_notaylor) < MIN_FLUX) &
                      (err_notaylor > np.log10(MACHINE_PRECISION))] = 1

    # Plot
    im1 = ax[0].imshow(err_starry, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect=aspect, cmap=errormap,
                       extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    im2 = ax[1].imshow(err_notaylor, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect=aspect, cmap=errormap,
                       extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    # Masks
    cmap_mask = pl.get_cmap('Greys')
    cmap_mask.set_under(alpha=0)
    cmap_mask.set_over(errormap(0))
    ax[0].imshow(err_starry_mask, origin='lower', aspect=aspect,
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax), 0, 2))
    ax[1].imshow(err_notaylor_mask, origin='lower', aspect=aspect,
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    # Appearance
    for axis in ax:
        axis.set_xlabel(r'$\log\,r$', fontsize=10, labelpad=0)
        axis.set_xlim(np.log10(rmin), np.log10(rmax))
        axis.set_ylim(0, 2)
        axis.set_ylabel(r'$b$', fontsize=10)
        axis.get_yaxis().set_label_coords(-0.175, 0.5)
        if eps == 0.1:
            axis.set_yticks([0, 1, 2])
            axis.set_yticklabels([r"$r + 0.9$", r"$r + 1$", r"$r + 1.1$"])
        else:
            axis.set_yticks([0, 1, 2])
            axis.set_yticklabels([r"$r + 1 - \epsilon$", r"$r + 1$",
                                  r"$r + 1 + \epsilon$"])
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontsize(6)

    return im1, im2


def RMinusOne(ax, l, m, res=1, eps=0.1, aspect='auto'):
    """Plot the error for b ~ r - 1."""
    blen = int(101 * res)
    rlen = int(301 * res)
    rmin = 1e-5
    rmax = 1e3
    r = np.logspace(np.log10(rmin), np.log10(rmax), rlen)

    # Exact
    m128 = starry.Map(l)
    m128.use_mp = True
    m128[l, m] = 1

    # Starry
    mt = starry.Map(l)
    mt[l, m] = 1

    # Starry (no taylor)
    mnt = starry.Map(l)
    mnt.optimize = False
    mnt[l, m] = 1

    # Compute
    flux_128 = np.zeros((blen, rlen))
    flux_starry = np.zeros((blen, rlen))
    flux_notaylor = np.zeros((blen, rlen))
    for i, ro in enumerate(r):
        b = np.linspace(ro - 1 - eps, ro - 1 + eps, blen)
        # Ensure the unstable value is in the array
        b[len(b) // 2] = ro - 1
        flux_128[:, i] = np.array(m128.flux(xo=0, yo=b, ro=ro))
        flux_starry[:, i] = np.array(mt.flux(xo=0, yo=b, ro=ro))
        flux_notaylor[:, i] = np.array(mnt.flux(xo=0, yo=b, ro=ro))

    # Compute the fractional errors
    err_starry = (flux_starry - flux_128) / (flux_128)
    err_starry[(flux_128 == 0) & (flux_starry == 0)] = MACHINE_PRECISION
    err_starry[(flux_128 == flux_starry)] = MACHINE_PRECISION
    err_starry = np.log10(np.abs(err_starry))

    err_notaylor = (flux_notaylor - flux_128) / flux_128
    err_notaylor[(flux_128 == 0) & (flux_notaylor == 0)] = MACHINE_PRECISION
    err_notaylor[(flux_128 == flux_notaylor)] = MACHINE_PRECISION
    err_notaylor = np.log10(np.abs(err_notaylor))

    # Mask regions where the flux is very small
    err_starry_mask = np.zeros_like(err_starry)
    err_starry_mask[(np.abs(flux_128) < MIN_FLUX) &
                    (np.abs(flux_starry) < MIN_FLUX) &
                    (err_starry > np.log10(MACHINE_PRECISION))] = 1
    err_notaylor_mask = np.zeros_like(err_notaylor)
    err_notaylor_mask[(np.abs(flux_128) < MIN_FLUX) &
                      (np.abs(flux_notaylor) < MIN_FLUX) &
                      (err_notaylor > np.log10(MACHINE_PRECISION))] = 1

    # Plot
    im1 = ax[0].imshow(err_starry, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect=aspect, cmap=errormap,
                       extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    im2 = ax[1].imshow(err_notaylor, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0, aspect=aspect, cmap=errormap,
                       extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    # Masks
    cmap_mask = pl.get_cmap('Greys')
    cmap_mask.set_under(alpha=0)
    cmap_mask.set_over(errormap(0))
    ax[0].imshow(err_starry_mask, origin='lower', aspect=aspect,
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax), 0, 2))
    ax[1].imshow(err_notaylor_mask, origin='lower', aspect=aspect,
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax), 0, 2))

    # Appearance
    for axis in ax:
        axis.set_xlabel(r'$\log\,r$', fontsize=10, labelpad=0)
        axis.set_xlim(np.log10(rmin), np.log10(rmax))
        axis.set_ylim(0, 2)
        axis.set_ylabel(r'$b$', fontsize=10)
        axis.get_yaxis().set_label_coords(-0.175, 0.5)
        if eps == 0.1:
            axis.set_yticks([0, 1, 2])
            axis.set_yticklabels([r"$r - 1.1$", r"$r - 1$", r"$r - 0.9$"])
        else:
            axis.set_yticks([0, 1, 2])
            axis.set_yticklabels([r"$r - 1 - \epsilon$", r"$r - 1$",
                                  r"$r - 1 + \epsilon$"])
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontsize(6)

    return im1, im2


def Ylm(l, m, res=1):
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

    ax1[0].set_title('Optimized')
    ax1[1].set_title('Original')
    im, _ = BRGrid(ax1, l, m, res=res)
    RMinusOneRPlusOne(ax2, l, m, res=res)
    RMinusOneRPlusOne(ax3, l, m, res=res, eps=1e-6)
    RPlusOne(ax4, l, m, res=res)
    RPlusOne(ax5, l, m, res=res, eps=1e-6)
    RMinusOne(ax6, l, m, res=res)
    RMinusOne(ax7, l, m, res=res, eps=1e-6)

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
    fig.savefig("Y_{%d,%d}.pdf" % (l, m))
    pl.close()


if __name__ == "__main__":
    lmax = 4
    for l in range(lmax + 1):
        print("Running l = %d..." % l)
        for m in tqdm(range(-l, l + 1)):
            Ylm(l, m, res=0.25)
