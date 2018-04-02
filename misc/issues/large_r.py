"""Numerical stability for large occultors."""
from starry import Map
import matplotlib.pyplot as pl
from tqdm import tqdm
import numpy as np
np.seterr(invalid='ignore', divide='ignore')
cmap = pl.get_cmap('plasma')


def Experiments():
    """Just playing around with stuff here."""
    # Knobs
    l = 10
    eps = 1e-6
    tol = 1e-6
    yo = 0
    npts = 50
    rarr = (np.logspace(-3, 3, npts))

    # Standard map
    ylm = Map(l)
    ylm._r_M = np.inf
    ylm._r_quartic = np.inf
    ylm._b_J = 0

    # Taylor expand `M` (odd nu)
    # and approximate occultor limb as parabola (even nu)
    ylmtaylor = Map(l)
    ylmtaylor._r_M = 0
    ylmtaylor._r_quartic = 0
    ylmtaylor._b_J = 0

    # Multiprecision (~exact)
    ylm128 = Map(l)
    ylm128.use_mp = True

    # Set up
    fig, ax = pl.subplots(2, 2, figsize=(12, 5))
    fig.suptitle("Degree $l = %d$" % l, fontsize=14)

    #  Loop over the orders
    noise = np.zeros((2 * l + 1, npts)) * np.nan
    error = np.zeros((2 * l + 1, npts)) * np.nan
    noisetaylor = np.zeros((2 * l + 1, npts)) * np.nan
    errortaylor = np.zeros((2 * l + 1, npts)) * np.nan
    for j, m in tqdm(enumerate(range(-l, l + 1)), total=2 * l + 1):
        # Set this coefficient
        parity = (l + m) % 2
        ylm.reset()
        ylm[l, m] = 1
        ylmtaylor.reset()
        ylmtaylor[l, m] = 1
        ylm128.reset()
        ylm128[l, m] = 1
        # Make sure we didn't set the s2 term,
        # which I still haven't fixed
        #if (l > 0) and (ylm.g[2] != 0):
        #    continue

        # Occultor radius loop
        for i, ro in enumerate(rarr):
            xo0 = 0.5 * ((ro + 1) + np.abs(ro - 1))
            xo = np.linspace(xo0 - 25 * eps, xo0 + 25 * eps, 50)
            flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
            fluxtaylor = np.array(ylmtaylor.flux(xo=xo, yo=yo, ro=ro) /
                                  (2 * np.sqrt(np.pi)))
            flux128 = np.array(ylm128.flux(xo=xo, yo=yo, ro=ro) /
                               (2 * np.sqrt(np.pi)))
            if np.any(flux128):
                slope = (flux[-1] - flux[0]) / (xo[-1] - xo[0])
                noise[j, i] = np.std(flux - (xo - xo[0]) * slope) / \
                              np.nanmedian(np.abs(flux128))
                error[j, i] = np.max(np.abs((flux / flux128 - 1)))
                slope = (fluxtaylor[-1] - fluxtaylor[0]) / (xo[-1] - xo[0])
                noisetaylor[j, i] = np.std(fluxtaylor - (xo - xo[0]) * slope) / \
                                    np.nanmedian(np.abs(flux128))
                errortaylor[j, i] = np.max(np.abs((fluxtaylor / flux128 - 1)))

    # Standard
    noise_even = np.nanmax(noise[::2], axis=0)
    error_even = np.nanmax(error[::2], axis=0)
    ax[0, 0].plot(rarr, noise_even, '-', lw=1, color="C0")
    ax[1, 0].plot(rarr, error_even, '-', lw=1, color="C0")
    if l > 0:
        noise_odd = np.nanmax(noise[1::2], axis=0)
        error_odd = np.nanmax(error[1::2], axis=0)
        ax[0, 1].plot(rarr, noise_odd, '-', lw=1, color="C1")
        ax[1, 1].plot(rarr, error_odd, '-', lw=1, color="C1")

    # Taylor
    noise_even = np.nanmax(noisetaylor[::2], axis=0)
    error_even = np.nanmax(errortaylor[::2], axis=0)
    ax[0, 0].plot(rarr, noise_even, '--', lw=1, color="C0")
    ax[1, 0].plot(rarr, error_even, '--', lw=1, color="C0")
    if l > 0:
        noise_odd = np.nanmax(noisetaylor[1::2], axis=0)
        error_odd = np.nanmax(errortaylor[1::2], axis=0)
        ax[0, 1].plot(rarr, noise_odd, '--', lw=1, color="C1")
        ax[1, 1].plot(rarr, error_odd, '--', lw=1, color="C1")

    # Appearance
    for axis in ax.flatten():
        axis.axhline(tol, color='k', ls='--', alpha=0.5, lw=0.5)
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_ylim(1e-17, 1e3)
        axis.set_xlim(1e-3, 1e3)
    ax[0, 0].set_ylabel("Fractional noise", fontsize=16)
    ax[1, 0].set_ylabel("Fractional error", fontsize=16)
    ax[1, 0].set_xlabel("Occultor radius", fontsize=16)
    ax[1, 1].set_xlabel("Occultor radius", fontsize=16)
    pl.show()


# Knobs
lmax = 10
eps = 1e-6
tol = 1e-6
yo = 0
npts = 50
rarr = (np.logspace(-3, 3, npts))

# Standard map
ylm = Map(lmax)
ylm._r_M = np.inf
ylm._r_quartic = np.inf * np.ones(11)
ylm._b_J = 0

# Taylor expand `M` (odd nu)
# and approximate occultor limb as parabola (even nu)
ylmtaylor = Map(lmax)
ylmtaylor._r_M = 0
ylmtaylor._r_quartic = np.zeros(11)
ylmtaylor._b_J = 0

# Variable thresholds (we will implement this one!)
ylmvar = Map(lmax)
ylmvar._r_M = 2
ylmvar._r_quartic = [100, 30, 30, 20, 15, 10, 8, 6, 5, 4, 3.5]
ylmvar._b_J = 0

# Multiprecision (~exact)
ylm128 = Map(lmax)
ylm128.use_mp = True

# Set up
fig, ax = pl.subplots(lmax + 1, 2, figsize=(4, 9))
fig.subplots_adjust(wspace=0, hspace=0, bottom=0.05, top=0.95)
ax[0, 0].set_title(r"$\nu$ even")
ax[0, 1].set_title(r"$\nu$ odd")

# Loop over the degrees
for l in range(lmax + 1):

    error = np.zeros((2 * l + 1, npts)) * np.nan
    errortaylor = np.zeros((2 * l + 1, npts)) * np.nan
    errorvar = np.zeros((2 * l + 1, npts)) * np.nan

    #  Loop over the orders
    for j, m in tqdm(enumerate(range(-l, l + 1)), total=2 * l + 1):
        # Set this coefficient
        parity = (l + m) % 2
        ylm.reset()
        ylm[l, m] = 1
        ylmtaylor.reset()
        ylmtaylor[l, m] = 1
        ylmvar.reset()
        ylmvar[l, m] = 1
        ylm128.reset()
        ylm128[l, m] = 1

        # Occultor radius loop
        for i, ro in enumerate(rarr):
            xo0 = 0.5 * ((ro + 1) + np.abs(ro - 1))
            xo = np.linspace(xo0 - 25 * eps, xo0 + 25 * eps, 50)
            flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
            fluxtaylor = np.array(ylmtaylor.flux(xo=xo, yo=yo, ro=ro) /
                                  (2 * np.sqrt(np.pi)))
            fluxvar = np.array(ylmvar.flux(xo=xo, yo=yo, ro=ro) /
                               (2 * np.sqrt(np.pi)))
            flux128 = np.array(ylm128.flux(xo=xo, yo=yo, ro=ro) /
                               (2 * np.sqrt(np.pi)))
            if np.any(flux128):
                error[j, i] = np.max(np.abs((flux / flux128 - 1)))
                errortaylor[j, i] = np.max(np.abs((fluxtaylor / flux128 - 1)))
                errorvar[j, i] = np.max(np.abs((fluxvar / flux128 - 1)))

    # Standard
    error_even = np.nanmax(error[::2], axis=0)
    ax[l, 0].plot(rarr, error_even, '-', lw=1, color="C0")
    if l > 0:
        error_odd = np.nanmax(error[1::2], axis=0)
        ax[l, 1].plot(rarr, error_odd, '-', lw=1, color="C0")

    # Taylor
    error_even = np.nanmax(errortaylor[::2], axis=0)
    ax[l, 0].plot(rarr, error_even, '--', lw=1, color="C0")
    if l > 0:
        error_odd = np.nanmax(errortaylor[1::2], axis=0)
        ax[l, 1].plot(rarr, error_odd, '--', lw=1, color="C0")

    # Variable
    error_even = np.nanmax(errorvar[::2], axis=0)
    ax[l, 0].plot(rarr, error_even, '-', lw=1, color="C1")
    if l > 0:
        error_odd = np.nanmax(errorvar[1::2], axis=0)
        ax[l, 1].plot(rarr, error_odd, '-', lw=1, color="C1")

    ax[l, 0].set_ylabel(r"$l$ = %d" % l, fontsize=8)

# Appearance
for axis in ax.flatten():
    axis.axhline(tol, color='k', ls='--', alpha=0.5, lw=0.5)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_ylim(1e-17, 1e3)
    axis.set_xlim(1e-3, 1e3)
pl.show()
