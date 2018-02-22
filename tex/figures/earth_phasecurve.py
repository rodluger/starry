"""Earth phase curve example."""
from starry import starry
import matplotlib.pyplot as pl
import numpy as np

# Generate a sample starry map
y = starry(10, 'earth')
y.rotate([0, 1, 0], -np.pi)

# Compute the phase curve
npts = 100
fig, ax = pl.subplots(1, figsize=(9, 9))
theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
F = y.flux(res=300, u=[0, 1, 0], theta=theta)
ax.plot(theta, F)

# Compute the phase curves for each continent
for continent in ['northamerica', 'southamerica',
                  'europe', 'asia', 'oceania',
                  'antarctica', 'africa']:
    y = starry(10, continent)
    y.rotate([0, 1, 0], -np.pi)
    F = y.flux(res=300, u=[0, 1, 0], theta=theta)
    ax.plot(theta, F)

# Save
pl.savefig('earth_phasecurve.pdf', bbox_inches='tight')
