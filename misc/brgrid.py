"""Stability in the b-r plane for linear limb darkening (Mandel & Agol)."""
import matplotlib.pyplot as pl
import starry
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

bmin = 1e-5
bmax = 1e3
blen = 301
rmin = 1e-5
rmax = 1e3
rlen = 301
b = np.logspace(np.log10(bmin), np.log10(bmax), blen)
r = np.logspace(np.log10(rmin), np.log10(rmax), rlen)
R, B = np.meshgrid(r, b)
MACHINE_PRECISION = 1.6e-16
MIN_FLUX = 1.e-9

# Exact
m128 = starry.Map(7)
m128.use_mp = True
m128[5, 4] = 1

# Starry
m = starry.Map(7)
m[5, 4] = 1

# Starry (no taylor)
mnt = starry.Map(7)
mnt.taylor = False
mnt[5, 4] = 1

# Compute
flux_128 = np.array(m128.flux(xo=0, yo=B, ro=R))
flux_starry = np.array(m.flux(xo=0, yo=B, ro=R))
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
vmax = 0
fig, ax = pl.subplots(2, 2, figsize=(8, 8))
im = ax[0, 0].imshow(err_starry, origin='lower',
                     vmin=np.log10(MACHINE_PRECISION),
                     vmax=vmax,
                     extent=(np.log10(rmin), np.log10(rmax),
                             np.log10(bmin), np.log10(bmax)))

im = ax[0, 1].imshow(err_notaylor, origin='lower',
                     vmin=np.log10(MACHINE_PRECISION),
                     vmax=vmax,
                     extent=(np.log10(rmin), np.log10(rmax),
                             np.log10(bmin), np.log10(bmax)))
axc = pl.axes([0.815, 0.1, 0.125, 0.8])
axc.axis('off')
pl.colorbar(im, label=r'$\log\,\mathrm{error}$')

# Masks
cmap_mask = pl.get_cmap('Greys')
cmap_mask.set_under(alpha=0)
cmap_mask.set_over((0.267004, 0.004874, 0.329415, 1))
ax[0, 0].imshow(err_starry_mask, origin='lower',
                vmin=0.4, vmax=0.6, cmap=cmap_mask,
                extent=(np.log10(rmin), np.log10(rmax),
                        np.log10(bmin), np.log10(bmax)))
ax[0, 1].imshow(err_notaylor_mask, origin='lower',
                vmin=0.4, vmax=0.6, cmap=cmap_mask,
                extent=(np.log10(rmin), np.log10(rmax),
                        np.log10(bmin), np.log10(bmax)))

# Appearance
for axis in (ax[0, 0], ax[0, 1]):
    axis.set_xlabel(r'$\log\,r$', fontsize=14)
    axis.set_ylabel(r'$\log\,b$', fontsize=14)
    axis.set_xlim(np.log10(rmin), np.log10(rmax))
    axis.set_ylim(np.log10(bmin), np.log10(bmax))
    r = np.logspace(np.log10(rmin), np.log10(rmax), 10000)
    axis.plot(np.log10(r), np.log10(1 - r), ls='--',
              lw=1, color='w', alpha=0.25)
    axis.plot(np.log10(r), np.log10(r - 1), ls='--',
              lw=1, color='w', alpha=0.25)
    axis.plot(np.log10(r), np.log10(1 + r), ls='--',
              lw=1, color='w', alpha=0.25)

    axis.plot(np.log10(r), np.log10(0.1 * (1 - r)), ls='--',
              lw=1, color='w', alpha=0.25)

pl.show()
