"""Starry speed tests."""
from starry import Star, Planet, System
import time
import matplotlib.pyplot as pl
import numpy as np
import batman

# Input params
u1 = 0.4
u2 = 0.26
mstar = 1       # solar masses
rstar = 1       # solar radii
rplanet = 0.1   # fraction of stellar radius
b0 = 0.5        # impact parameter
P = 50          # orbital period in days

# Compute the semi-major axis from Kepler's
# third law in units of rstar (for batman)
a = ((P * 86400) ** 2 * (1.32712440018e20 * mstar) /
     (4 * np.pi ** 2)) ** (1. / 3.) / (6.957e8 * rstar)

# Get the inclination in degrees
inc = np.arccos(b0 / a) * 180 / np.pi

# Starry expects the planet radius in units of REARTH:
r = rplanet * 6.957e8 * rstar / 6.3781e6

# Timing params
number = 10
nN = 8
Nmax = 5
Narr = np.logspace(1, Nmax, nN)
starry_time = np.zeros(nN)
batman_time = np.zeros(nN)

# Loop over number of cadences
for i, N in enumerate(Narr):

    # Time array
    t = np.linspace(-1, 1, N)

    # starry
    star = Star(m=mstar, r=rstar, L=1)
    star.map.limbdark(u1, u2)
    planet = Planet(r=r, inc=inc, porb=P, lambda0=90)
    system = System([star, planet])
    tstart = time.time()
    for k in range(number):
        system.compute(t)
    starry_time[i] = time.time() - tstart

    # batman
    params = batman.TransitParams()
    params.limb_dark = "quadratic"
    params.u = [u1, u2]
    params.t0 = 0.
    params.ecc = 0
    params.w = 90.
    params.rp = rplanet
    params.a = a
    params.per = P
    params.inc = inc
    m = batman.TransitModel(params, t, nthreads=1)
    tstart = time.time()
    for k in range(number):
        # HACK: batman pre-computes the orbital solution when
        # TransitModel() is called. This is useful when the user
        # wants to compute several light curves given the exact same
        # orbits -- say, changing only the limb darkening parameters or
        # the planet size. I could implement the same in
        # starry, but I don't think this is particularly useful when
        # fitting a photodynamical model to data, since in reality all
        # orbital parameters are free parameters. In order to compare
        # apples to apples, let's clear the batman cache and force it to
        # re-compute the orbital solution by setting m.t0 = None:
        m.t0 = None
        m.light_curve(params)
    batman_time[i] = time.time() - tstart

# Plot
fig, ax = pl.subplots(1, figsize=(4, 3))
ax.plot(Narr, starry_time, 'o', ms=2, color='C0')
ax.plot(Narr, starry_time, '-', lw=0.5, color='C0', label='starry')
ax.plot(Narr, batman_time, 'o', ms=2, color='C1')
ax.plot(Narr, batman_time, '-', lw=0.5, color='C1', label='batman')

# Tweak and save
ax.legend(fontsize=9, loc='upper left')
ax.set_ylabel("Time [s]", fontsize=10)
ax.set_xlabel("Number of points", fontsize=10)
ax.set_xscale('log')
ax.set_yscale('log')

# Print average ratio
print(np.nanmedian(starry_time / batman_time))
fig.savefig("speed_batman.pdf", bbox_inches='tight')
