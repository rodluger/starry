"""Numerical stability for large occultors."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np
cmap = pl.get_cmap('plasma')


def color(l):
    """Return the color for the spherical harmonic degree `l`."""
    return cmap(0.1 + 0.8 * l / lmax)


# Knobs
lmax = 10
niter = 10
eps = 1e-6
tol = 1e-6
rarr = np.concatenate((np.logspace(-3, 0, 10, endpoint=False),
                       np.logspace(0, 3, 20)))
noise = np.zeros((lmax + 1, len(rarr)))
yo = 0
ylm = Map(lmax)

# Set up
fig, ax = pl.subplots(1, figsize=(9, 4))

# Loop over the degrees
for l in range(lmax + 1):
    ylm.reset()
    # Set the coefficients for all orders
    for m in range(-l, l + 1):
        ylm[l, m] = 1
    # Occultor radius loop
    for i, ro in enumerate(rarr):
        # Do this a few times and take the average
        for j in range(niter):
            # Compute the standard deviation on the flux
            # in a very tiny region mid-ingress
            r1 = ro - 0.5 + 0.1 * np.random.random()
            r2 = r1 + 50 * eps
            xo = np.linspace(r1, r2, 50)
            flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
            slope = (flux[-1] - flux[0]) / (r2 - r1)
            flux -= (xo - r1) * slope
            noise[l, i] += np.std(flux)
        noise[l, i] /= niter
    ax.plot(rarr, noise[l], '-', color=color(l), lw=1, label=r"$l=%d$" % l)

    # Figure out the radius at which the noise exceeds the tolerance
    if np.any(noise[l] > tol):
        maxr = np.argmax(noise[l] > tol)
        y2 = np.log10(noise[l, maxr])
        y1 = np.log10(noise[l, maxr - 1])
        x2 = np.log10(rarr[maxr])
        x1 = np.log10(rarr[maxr - 1])
        m = (y2 - y1) / (x2 - x1)
        y = np.log10(tol)
        x = (x1 + (y - y1) / m)
        ax.plot([10 ** x, 10 ** x], [5e-13, tol], color='k',
                alpha=0.5, ls='-', lw=0.5)
        ax.plot(10 ** x, tol, '.', ms=4, color=color(l))

        ax.annotate("%d" % l, xy=(10 ** x, 1e-15),
                    xycoords="data", xytext=(0, 25),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=8, arrowprops=dict(arrowstyle="-|>", lw=0.5))

ax.legend(loc="upper left", ncol=2)
ax.axhline(tol, color='k', ls='--', alpha=0.5, lw=0.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-15, 5.)
ax.set_xlim(1e-3, 1e3)
ax.set_xlabel("Occultor radius", fontsize=16)
ax.set_ylabel("Fractional error", fontsize=16)
fig.savefig("stability.pdf", bbox_inches='tight')
