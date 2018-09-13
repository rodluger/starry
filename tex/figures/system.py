"""Exoplanet system example."""
from starry.kepler import Primary, Secondary, System
import matplotlib.pyplot as pl
import numpy as np


# Setup
fig, ax = pl.subplots(2, figsize=(12, 6))
fig.subplots_adjust(hspace=0.35)
ax[0].set_ylabel('Normalized flux', fontsize=16)
ax[0].set_xlabel('Time [days]', fontsize=16)
ax[1].set_ylabel('Normalized flux', fontsize=16)
ax[1].set_xlabel('Time [days]', fontsize=16)
ax[0].set_xlim(0, 20)
ax[1].set_xlim(0, 20)
time = np.linspace(0, 20, 10000)


# System #1: A one-planet system with a hotspot offset
# ----------------------------------------------------

# Instantiate the star
star = Primary()

# Give the star a quadratic limb darkening profile
star[1] = 0.4
star[2] = 0.26

# Instantiate planet b
b = Secondary()
b.r = 0.091679
b.L = 5e-3
b.inc = 90
b.porb = 4.3
b.prot = 4.3
b.a = 11.127991
b.lambda0 = 90
b.tref = 2
b.axis = [0, 1, 0]

# Give the planet a simple dipole map
b[1, 0] = 0.5

# Rotate the planet map to produce a hotspot offset of 15 degrees
b.rotate(theta=15)

# Compute and plot the starry flux
time = np.linspace(0, 20, 10000)
system = System(star, b)
system.compute(time)
sF = np.array(system.lightcurve)
sF /= sF[0]
ax[0].plot(time, sF, '-', color='C0')


# System #2: A three-planet system with planet-planet occultations
# ----------------------------------------------------------------

# Instantiate the star
star = Primary()

# Give the star a quadratic limb darkening profile
star[1] = 0.4
star[2] = 0.26

# Instantiate planet b
b = Secondary()
b.r = 0.04584
b.L = 1e-3
b.inc = 89.0
b.porb = 2.1
b.prot = 2.1
b.a = 6.901084
b.lambda0 = 90
b.tref = 0.5

# Give the planet a wonky map
b[1, 0] = 0.5
b[2, 1] = 0.5

# Instantiate planet c
c = Secondary()
c.r = 0.07334
c.L = 1e-3
c.inc = 90.0
c.porb = 6.7
c.prot = 6.7
c.a = 14.95619
c.lambda0 = 90
c.tref = 3
c.axis = [0, 1, 0]

# Give the planet a wonky map
c[1, 0] = 0.5
c[2, -2] = 0.5

# Rotate the planet map to produce a hotspot offset of 15 degrees
c.rotate(theta=15)

# Compute and plot the starry flux
system = System(star, b, c)
system.compute(time)
sF = np.array(system.lightcurve)
sF /= sF[0]
ax[1].plot(time, sF, '-', color='C0')

# Save
for tick in ax[0].get_xticklabels() + ax[1].get_xticklabels():
    tick.set_fontsize(14)
pl.savefig('system.pdf', bbox_inches='tight')
