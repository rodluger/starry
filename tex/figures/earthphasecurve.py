"""Earth phase curve example."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np

# Set up the plot
npts = 100
nptsnum = 10
fig, ax = pl.subplots(1, figsize=(12, 4))
theta = np.linspace(0, 2 * np.pi, npts, endpoint=True)
thetanum = np.linspace(0, 2 * np.pi, nptsnum, endpoint=True)
total = np.zeros(npts, dtype=float)

# Compute the phase curves for each continent
base = 0.65
continents = ['asia.jpg', 'africa.jpg',
              'southamerica.jpg', 'northamerica.jpg', 'oceania.jpg',
              'europe.jpg', 'antarctica.jpg']
labels = ['Asia', 'Africa', 'S. America',
          'N. America', 'Oceania',
          'Europe',
          'Antarctica']
m = Map(10)
for continent, label in zip(continents, labels):
    m.load_image(continent)
    m.rotate([0, 1, 0], -np.pi)
    F = m.flux(u=[0, 1, 0], theta=theta)
    ax.plot(theta * 180 / np.pi - 180, F - base, label=label)

# Compute and plot the total phase curve
m.load_image('earth.jpg')
m.rotate([0, 1, 0], -np.pi)
total = m.flux(u=[0, 1, 0], theta=theta)
ax.plot(theta * 180 / np.pi - 180, total - base, 'k-', label='Total')

# Compute and plot the total phase curve (numerical)
totalnum = m.flux(u=[0, 1, 0], theta=thetanum, numerical=True, tol=1e-5)
ax.plot(thetanum * 180 / np.pi - 180, totalnum - base, 'k.')

# Appearance
ax.set_xlim(-180, 180)
ax.set_ylim(-0.05, 1.1)
ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc='best', fontsize=12, ncol=2)
ax.set_xlabel('Sub-observer longitude [deg]', fontsize=24)
ax.set_ylabel('Normalized flux', fontsize=24)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontsize(22)

# Save
pl.savefig('earthphasecurve.pdf', bbox_inches='tight')
