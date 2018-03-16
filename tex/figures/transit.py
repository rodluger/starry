"""Stellar transit example."""
from starry import Map
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


# Instantiate the star
u1 = 0.4
u2 = 0.26
m = Map(2)
m.limbdark(u1, u2)

# Occultor (planet) params
ro = 0.1
npts = 500
time = np.linspace(-1, 1, npts)
b0 = 0.5
P = 50.
a = 15.

# Orbital solution for zero eccentricity
inc = np.arccos(b0 / a)
f = 2 * np.pi / P * time
xo = a * np.cos(np.pi / 2. + f)
b = a * np.sqrt(1 - np.sin(np.pi / 2. + f) ** 2 * np.sin(inc) ** 2)
yo = np.sqrt(b ** 2 - xo ** 2)

# Compute and plot the starry flux
print("Computing starry flux...")
fig, ax = pl.subplots(2, sharex=True)
fig.subplots_adjust(hspace=0.05)
sF = m.flux(xo=xo, yo=yo, ro=ro)
sF /= np.max(sF)
ax[0].plot(time, sF, '-', color='C0', label='starry')

# Compute and plot the flux from numerical integration
print("Computing numerical flux...")
nF = np.zeros_like(time)
for i in tqdm(range(npts)):
    nF[i] = NumericalFlux(b[i], ro, u1, u2)
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
params.rp = ro
params.a = a
params.per = P
params.inc = inc * 180 / np.pi
m = batman.TransitModel(params, time)
bF = m.light_curve(params)
ax[0].plot(time[::20], bF[::20], '.', color='C1', label='batman')

# Plot the residual error
ax[1].plot(time, 1e6 * np.abs(nF - sF) / (1 - nF), '-',
           color='C0', label='starry')
ax[1].plot(time, 1e6 * np.abs(nF - bF) / (1 - nF), '-',
           color='C1', label='batman')
ax[1].set_yscale('log')

# Appearance
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
ax[0].set_xlim(-1, 1)
ax[0].set_ylabel('Normalized flux', fontsize=16)
ax[1].set_ylabel('Depth error [ppm]', fontsize=16, labelpad=10)
ax[1].set_xlabel('Time [arbitrary units]', fontsize=16)

# Save
pl.savefig('transit.pdf', bbox_inches='tight')
