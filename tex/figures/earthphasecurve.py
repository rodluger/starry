"""Earth phase curve example."""
from starry import starry
import matplotlib.pyplot as pl
import numpy as np

# Set up the plot
npts = 100
fig, ax = pl.subplots(1, figsize=(12, 5))
theta = np.linspace(0, 2 * np.pi, npts, endpoint=True)
total = np.zeros(npts, dtype=float)

# Compute the phase curves for each continent
base = 116.5
norm = 180
continents = ['asia', 'africa',
              'southamerica', 'northamerica', 'oceania',
              'europe', 'antarctica']
labels = [r'$\mathrm{Asia}$', r'$\mathrm{Africa}$', r'$\mathrm{S. America}$',
          r'$\mathrm{N. America}$', r'$\mathrm{Oceania}$',
          r'$\mathrm{Europe}$',
          r'$\mathrm{Antarctica}$']
for continent, label in zip(continents, labels):
    y = starry(10, continent)
    y.rotate([0, 1, 0], -np.pi)
    F = y.flux(res=300, u=[0, 1, 0], theta=theta)
    F = (F - base) / norm
    total += F
    ax.plot(theta * 180 / np.pi - 180, F, label=label)

# Plot the total phase curve
ax.plot(theta * 180 / np.pi - 180, total, 'k-', label=r'$\mathrm{Total}$')

# Appearance
ax.set_xlim(-180, 180)
ax.set_ylim(-0.01, 1.1)
ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
ax.set_xticklabels([r'$-180$', r'$-135$', r'$-90$', r'$-45$',
                    r'$0$', r'$45$', r'$90$', r'$135$', r'$180$'])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([r'$0.0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1.0$'])
ax.legend(loc='best', fontsize=14, ncol=2)
ax.set_xlabel(r'$\mathrm{Sub}-\mathrm{observer\ longitude\ (deg)}$',
              fontsize=24)
ax.set_ylabel(r'$\mathrm{Normalized\ flux}$', fontsize=24)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontsize(22)

# Save
pl.savefig('earthphasecurve.pdf', bbox_inches='tight')
