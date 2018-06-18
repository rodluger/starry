"""Stellar transit example."""
from starry import Star, Planet, System
from tqdm import tqdm
import matplotlib.pyplot as pl
import numpy as np
import batman
from scipy.integrate import dblquad


def NumericalFlux(b, r, u1, u2):
    """Compute the flux by numerical integration of the surface integral."""
    # I'm only coding up a specific case here
    assert (b >= 0) and (r <= 1), "Invalid range."

    # Total flux
    total = (np.pi / 6) * (6 - 2 * u1 - u2)
    if b >= 1 + r:
        return total

    # Get points of intersection
    if b > 1 - r:
        yi = (1. + b ** 2 - r ** 2) / (2. * b)
        xi = (1. / (2. * b)) * np.sqrt(4 * b ** 2 - (1 + b ** 2 - r ** 2) ** 2)
    else:
        yi = np.inf
        xi = r

    # Specific intensity map
    def I(y, x):
        mu = np.sqrt(1 - x ** 2 - y ** 2)
        return 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2

    # Lower integration limit
    def y1(x):
        if yi <= b:
            # Lower occultor boundary
            return b - np.sqrt(r ** 2 - x ** 2)
        elif b <= 1 - r:
            # Lower occultor boundary
            return b - np.sqrt(r ** 2 - x ** 2)
        else:
            # Tricky: we need to do this in two parts
            return b - np.sqrt(r ** 2 - x ** 2)

    # Upper integration limit
    def y2(x):
        if yi <= b:
            # Upper occulted boundary
            return np.sqrt(1 - x ** 2)
        elif b <= 1 - r:
            # Upper occultor boundary
            return b + np.sqrt(r ** 2 - x ** 2)
        else:
            # Tricky: we need to do this in two parts
            return np.sqrt(1 - x ** 2)

    # Compute the total flux
    flux, _ = dblquad(I, -xi, xi, y1, y2, epsabs=1e-14, epsrel=1e-14)

    # Do we need to solve an additional integral?
    if not (yi <= b) and not (b <= 1 - r):

        def y1(x):
            return b - np.sqrt(r ** 2 - x ** 2)

        def y2(x):
            return b + np.sqrt(r ** 2 - x ** 2)

        additional_flux, _ = dblquad(I, -r, -xi, y1, y2,
                                     epsabs=1e-14, epsrel=1e-14)

        flux += 2 * additional_flux

    return total - flux


# Setup
fig, ax = pl.subplots(2, sharex=True)
fig.subplots_adjust(hspace=0.05)

# Input params
u1 = 0.4
u2 = 0.26
mstar = 1       # solar masses
rstar = 1       # solar radii
rplanet = 0.1   # fraction of stellar radius
b0 = 0.5        # impact parameter
P = 50          # orbital period in days
npts = 500
time = np.linspace(-0.25, 0.25, npts)

# Compute the semi-major axis from Kepler's third law in units of rstar
a = ((P * 86400) ** 2 * (1.32712440018e20 * mstar) /
     (4 * np.pi ** 2)) ** (1. / 3.) / (6.957e8 * rstar)

# Get the inclination in degrees
inc = np.arccos(b0 / a) * 180 / np.pi

# Compute and plot the starry flux
star = Star()
star.map[1] = u1
star.map[2] = u2

planet = Planet(r=rplanet, inc=inc, porb=P, a=a, lambda0=90)
system = System([star, planet])
system.compute(time)
sF = np.array(star.flux)
sF /= sF[0]
ax[0].plot(time, sF, '-', color='C0', label='starry')

# Compute and plot the flux from numerical integration
print("Computing numerical flux...")
f = 2 * np.pi / P * time
b = a * np.sqrt(1 - np.sin(np.pi / 2. + f) ** 2
                * np.sin(inc * np.pi / 180) ** 2)
nF = np.zeros_like(time)
for i in tqdm(range(npts)):
    nF[i] = NumericalFlux(b[i], rplanet, u1, u2)
nF /= np.nanmax(nF)
ax[0].plot(time[::20], nF[::20], 'o', color='C4', label='numerical')

# Compute and plot the batman flux
print("Computing batman flux...")
params = batman.TransitParams()
params.limb_dark = "quadratic"
params.u = [u1, u2]
params.t0 = 0.
params.ecc = 0.
params.w = 90.
params.rp = rplanet
params.a = a
params.per = P
params.inc = inc
m = batman.TransitModel(params, time)
bF = m.light_curve(params)
ax[0].plot(time[::20], bF[::20], '.', color='C1', label='batman')

# Plot the relative error
sE = np.abs(nF - sF)
sE[sE == 0] = np.nan
bE = np.abs(nF - bF)
bE[bE == 0] = np.nan
ax[1].plot(time, sE, '-',
           color='C0', label='starry')
ax[1].plot(time, bE, '-',
           color='C1', label='batman')
ax[1].set_yscale('log')

# Appearance
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
ax[0].set_xlim(-0.25, 0.25)
ax[0].set_ylabel('Normalized flux', fontsize=16)
ax[1].set_ylabel('Relative error', fontsize=16, labelpad=10)
ax[1].set_xlabel('Time [arbitrary units]', fontsize=16)

# Save
pl.savefig('transit.pdf', bbox_inches='tight')
