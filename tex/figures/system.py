"""Exoplanet system example. TODO: Not yet finished."""
from starry import Star, Planet, System
import matplotlib.pyplot as pl
import numpy as np

# debug
pl.switch_backend('MacOSX')

# Setup
fig, ax = pl.subplots(1, figsize=(12, 5))

# Instantiate the star
star = Star(m=1, r=1, L=1)

# Give the star a quadratic limb darkening profile
star.map.limbdark(0.4, 0.26)

# Instantiate planet b
b = Planet(r=10,
           L=3e-3,
           inc=90,
           porb=3,
           prot=3,
           lambda0=90,
           tref=2)

# Give the planet a simple dipole map
b.map[0, 0] = 2
b.map[1, 0] = -1

# Rotate the planet map to produce a hotspot offset of 15 degrees
b.map.rotate(u=(0, 1, 0), theta=15 * np.pi / 180)

# Instantiate planet c
c = Planet(r=5,
           L=5e-4,
           inc=90,
           porb=10,
           prot=10,
           lambda0=90,
           tref=4)

# Give the planet a constant map for simplicity
c.map[0, 0] = 1

# Compute and plot the starry flux
time = np.linspace(0, 12, 10000)
system = System([star, b, c])
system.compute(time)
sF = np.array(system.flux)
sF /= sF[0]
ax.plot(time, sF, '-', color='C0')

# Appearance
ax.set_ylabel('Normalized flux', fontsize=16)
ax.set_xlabel('Time [days]', fontsize=16)

# Save
pl.savefig('system.pdf', bbox_inches='tight')
