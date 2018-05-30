"""Exoplanet system example."""
from starry import Star, Planet, System
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
star = Star()

# Give the star a quadratic limb darkening profile
star.map[1] = 0.4
star.map[2] = 0.26

# Instantiate planet b
b = Planet(r=0.091679,
           L=5e-3,
           inc=90,
           porb=4.3,
           prot=4.3,
           a=11.127991,
           lambda0=90,
           tref=2)

# Give the planet a simple dipole map
b.map[1, 0] = 0.5

# Rotate the planet map to produce a hotspot offset of 15 degrees
b.map.rotate(axis=(0, 1, 0), theta=15)

# Compute and plot the starry flux
time = np.linspace(0, 20, 10000)
system = System([star, b])
system.compute(time)
sF = np.array(system.flux)
sF /= sF[0]
ax[0].plot(time, sF, '-', color='C0')


# System #2: A three-planet system with planet-planet occultations
# ----------------------------------------------------------------

# Instantiate the star
star = Star()

# Give the star a quadratic limb darkening profile
star.map[1] = 0.4
star.map[2] = 0.26

# Instantiate planet b
b = Planet(r=0.04584,
           L=1e-3,
           inc=89.0,
           porb=2.1,
           prot=2.1,
           a=6.901084,
           lambda0=90,
           tref=0.5)

# Give the planet a wonky map
b.map[1, 0] = 0.5
b.map[2, 1] = 0.5

# Instantiate planet c
c = Planet(r=0.07334,
           L=1e-3,
           inc=90.0,
           porb=6.7,
           prot=6.7,
           a=14.95619,
           lambda0=90,
           tref=3)

# Give the planet a wonky map
c.map[1, 0] = 0.5
c.map[2, -2] = 0.5

# Rotate the planet map to produce a hotspot offset of 15 degrees
c.map.rotate(axis=(0, 1, 0), theta=15)

# Compute and plot the starry flux
system = System([star, b, c])
system.compute(time)
sF = np.array(system.flux)
sF /= sF[0]
ax[1].plot(time, sF, '-', color='C0')

# Save
for tick in ax[0].get_xticklabels() + ax[1].get_xticklabels():
    tick.set_fontsize(14)
pl.savefig('system.pdf', bbox_inches='tight')
