"""Earth phase curve example."""
from starry import starry
import matplotlib.pyplot as pl
import matplotlib
matplotlib.rc('text', usetex=True)
import numpy as np

# Set up the plot
npts = 100
fig, ax = pl.subplots(1, figsize=(12, 7))
theta = np.linspace(0, 2 * np.pi, npts, endpoint=True)
total = np.zeros(npts, dtype=float)

# Compute the phase curves for each continent
base = 116.5
norm = 180
continents = ['asia', 'africa',
              'southamerica', 'northamerica', 'oceania',
              'europe', 'antarctica']
labels = ['Asia', 'Africa', 'S. America',
          'N. America', 'Oceania', 'Europe',
          'Antactica']
for continent, label in zip(continents, labels):
    y = starry(10, continent)
    y.rotate([0, 1, 0], -np.pi)
    F = y.flux(res=300, u=[0, 1, 0], theta=theta)
    F = (F - base) / norm
    total += F
    ax.plot(theta * 180 / np.pi - 180, F, label=label)

# Plot the total phase curve
ax.plot(theta * 180 / np.pi - 180, total, 'k-', label='Total')

# Appearance
ax.set_xlim(-180, 180)
ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
ax.legend(loc='best', fontsize=12)
ax.set_xlabel('Sub-observer longitude (deg)', fontsize=18)
ax.set_ylabel('Normalized flux', fontsize=18)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontsize(15)

# Save
pl.savefig('earth_phasecurve.pdf', bbox_inches='tight')
