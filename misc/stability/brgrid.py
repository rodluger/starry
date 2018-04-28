"""Stability in the b-r plane for linear limb darkening (Mandel & Agol)."""
import matplotlib.pyplot as pl
import starry
import numpy as np
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')
MACHINE_PRECISION = 1.6e-16
MIN_FLUX = 1.e-9


def BRGrid(ax, l, m, res=1):
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
                       vmax=0,
                       extent=(np.log10(rmin), np.log10(rmax),
                               np.log10(bmin), np.log10(bmax)))

    im2 = ax[1].imshow(err_notaylor, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0,
                       extent=(np.log10(rmin), np.log10(rmax),
                               np.log10(bmin), np.log10(bmax)))

    # Masks
    cmap_mask = pl.get_cmap('Greys')
    cmap_mask.set_under(alpha=0)
    cmap_mask.set_over((0.267004, 0.004874, 0.329415, 1))
    ax[0].imshow(err_starry_mask, origin='lower',
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax),
                         np.log10(bmin), np.log10(bmax)))
    ax[1].imshow(err_notaylor_mask, origin='lower',
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax),
                         np.log10(bmin), np.log10(bmax)))

    # Appearance
    for axis in ax:
        axis.set_xlabel(r'$\log\,r$', fontsize=14)
        axis.set_xlim(np.log10(rmin), np.log10(rmax))
        axis.set_ylim(np.log10(bmin), np.log10(bmax))

        '''
        # Boundary lines
        r = np.logspace(np.log10(rmin), np.log10(rmax), 10000)
        axis.plot(np.log10(r), np.log10(1 - r), ls='--',
                  lw=1, color='w', alpha=0.25)
        axis.plot(np.log10(r), np.log10(r - 1), ls='--',
                  lw=1, color='w', alpha=0.25)
        axis.plot(np.log10(r), np.log10(1 + r), ls='--',
                  lw=1, color='w', alpha=0.25)
        axis.plot(np.log10(r), np.log10(0.1 * (1 - r)), ls='--',
                  lw=1, color='w', alpha=0.25)
        '''

    ax[0].set_ylabel(r'$\log\,b$', fontsize=14)

    return im1, im2


def LargeR(ax, l, m, res=1, deltab=1):
    """Plot the error for large radii."""
    blen = int(51 * res)
    rlen = int(301 * res)
    rmin = 1
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
        b = np.linspace(ro - deltab, ro + deltab, blen)
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
    im3 = ax[0].imshow(err_starry, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0,
                       extent=(np.log10(rmin), np.log10(rmax), 0, 1))

    im4 = ax[1].imshow(err_notaylor, origin='lower',
                       vmin=np.log10(MACHINE_PRECISION),
                       vmax=0,
                       extent=(np.log10(rmin), np.log10(rmax), 0, 1))

    # Masks
    cmap_mask = pl.get_cmap('Greys')
    cmap_mask.set_under(alpha=0)
    cmap_mask.set_over((0.267004, 0.004874, 0.329415, 1))
    ax[0].imshow(err_starry_mask, origin='lower',
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax), 0, 1))
    ax[1].imshow(err_notaylor_mask, origin='lower',
                 vmin=0.4, vmax=0.6, cmap=cmap_mask,
                 extent=(np.log10(rmin), np.log10(rmax), 0, 1))

    # Appearance
    for axis in ax:
        axis.set_xlabel(r'$\log\,r$', fontsize=14)
        axis.set_xlim(np.log10(rmin), np.log10(rmax))
        axis.set_ylim(0, 1)
        axis.set_yticks([0, 0.5, 1])
    ax[0].set_yticklabels(["Ing", "Mid", "Tot"])
    ax[1].set_yticklabels([])

    return im3, im4


def Ylm(l, m, res=1):
    """Generate a stability plot for a given Ylm."""
    fig = pl.figure(figsize=(9, 6))
    fig.subplots_adjust(right=0.8, top=0.925)
    axBR = [pl.subplot2grid([4, 2], [0, 0], rowspan=3),
            pl.subplot2grid([4, 2], [0, 1], rowspan=3)]
    axLR = [pl.subplot2grid([4, 2], [3, 0], rowspan=3),
            pl.subplot2grid([4, 2], [3, 1], rowspan=3)]
    axBR[0].set_title('Taylor')
    axBR[1].set_title('Original')
    im1, im2 = BRGrid(axBR, l, m, res=res)
    im3, im4 = LargeR(axLR, l, m, res=res, deltab=1e-12)
    axc = pl.axes([0.725, 0.075, 0.15, 0.8])
    axc.axis('off')
    cb = pl.colorbar(im1)
    cb.set_label(r'$\log\,\mathrm{error}$', fontsize=14)
    pl.suptitle(r"$Y_{%d,%d}$" % (l, m), fontsize=24, y=0.98)
    fig.savefig("Y_{%d,%d}.pdf" % (l, m), bbox_inches='tight')
    pl.close()


if __name__ == "__main__":

    # DEBUG
    Ylm(1, 0)
    quit()

    lmax = 4
    for l in range(lmax + 1):
        print("Running l = %d..." % l)
        for m in tqdm(range(-l, l + 1)):
            Ylm(l, m, res=0.25)
