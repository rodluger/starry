"""Exoplanet system example. TODO: Not yet finished."""
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
star = Star(m=1, r=1, L=1)

# Give the star a quadratic limb darkening profile
star.map.limbdark(0.4, 0.26)

# Instantiate planet c
b = Planet(r=10,
           L=5e-3,
           inc=90,
           porb=4.3,
           prot=4.3,
           lambda0=90,
           tref=2)

# Give the planet a simple dipole map
b.map[0, 0] = 2
b.map[1, 0] = -1

# Rotate the planet map to produce a hotspot offset of 15 degrees
b.map.rotate(u=(0, 1, 0), theta=15 * np.pi / 180)

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
star = Star(m=1, r=1, L=1)

# Give the star a quadratic limb darkening profile
star.map.limbdark(0.4, 0.26)

# Instantiate planet b
b = Planet(r=5,
           L=5e-3,
           inc=89.0,
           porb=2.1,
           prot=2.1,
           lambda0=90,
           tref=0.5)

# Give the planet a wonky map
b.map[0, 0] = 2
b.map[1, 0] = -1
b.map[2, 1] = 1

# Instantiate planet c
c = Planet(r=8,
           L=5e-3,
           inc=90.0,
           porb=6.7,
           prot=6.7,
           lambda0=90,
           tref=3)

# Give the planet a wonky map
c.map[0, 0] = 2
c.map[1, 0] = -1
c.map[2, -2] = 1

# Rotate the planet map to produce a hotspot offset of 15 degrees
c.map.rotate(u=(0, 1, 0), theta=15 * np.pi / 180)

# Compute and plot the starry flux
system = System([star, b, c])
system.compute(time)
sF = np.array(system.flux)
sF /= sF[0]
ax[1].plot(time, sF, '-', color='C0')

# Save
pl.savefig('system.pdf', bbox_inches='tight')
