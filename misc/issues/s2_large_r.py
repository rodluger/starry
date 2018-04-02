"""Numerical stability for large occultors."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np
np.seterr(invalid='ignore', divide='ignore')
cmap = pl.get_cmap('plasma')


# Knobs
eps = 1e-6
tol = 1e-6
yo = 0
npts = 50
rarr = (np.logspace(-3, 3, npts))

# Standard map
ylm = Map(1)
ylm[1, 0] = 1
ylm.taylor_r = np.inf
ylm.quad_r = np.inf
ylm.taylor_b = 0

# Multiprecision (~exact)
ylm128 = Map(1)
ylm128[1, 0] = 1
ylm128.use_mp = True

# Set up
fig, ax = pl.subplots(2, figsize=(4, 5))
fig.suptitle("s2 term", fontsize=14)

# Occultor radius loop
noise = np.zeros(npts)
error = np.zeros(npts)
for i, ro in enumerate(rarr):
    xo0 = 0.5 * ((ro + 1) + np.abs(ro - 1))
    xo = np.linspace(xo0 - 25 * eps, xo0 + 25 * eps, 50)
    flux = ylm.flux(xo=xo, yo=yo, ro=ro) / (2 * np.sqrt(np.pi))
    flux128 = np.array(ylm128.flux(xo=xo, yo=yo, ro=ro) /
                       (2 * np.sqrt(np.pi)))
    if np.any(flux128):
        slope = (flux[-1] - flux[0]) / (xo[-1] - xo[0])
        noise[i] = np.std(flux - (xo - xo[0]) * slope) / \
                   np.nanmedian(np.abs(flux128))
        error[i] = np.max(np.abs((flux / flux128 - 1)))

# Standard
ax[0].plot(rarr, noise, '-', lw=1, color="C0")
ax[1].plot(rarr, error, '-', lw=1, color="C0")

# Appearance
for axis in ax.flatten():
    axis.axhline(tol, color='k', ls='--', alpha=0.5, lw=0.5)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_ylim(1e-17, 1e3)
    axis.set_xlim(1e-3, 1e3)
ax[0].set_ylabel("Fractional noise", fontsize=16)
ax[1].set_ylabel("Fractional error", fontsize=16)
ax[1].set_xlabel("Occultor radius", fontsize=16)
pl.show()
