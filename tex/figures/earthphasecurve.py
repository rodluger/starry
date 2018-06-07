"""Earth phase curve example."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np

# Set up the plot
nim = 12
npts = 100
nptsnum = 10
fig = pl.figure(figsize=(12, 5))
ax = pl.subplot2grid((5, nim), (1, 0), colspan=nim, rowspan=4)
theta = np.linspace(0, 360, npts, endpoint=True)
thetanum = np.linspace(0, 360, nptsnum, endpoint=True)
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
    m.rotate([0, 1, 0], -180)
    F = m.flux(axis=[0, 1, 0], theta=theta)
    F -= np.nanmin(F)
    ax.plot(theta - 180, F, label=label)

# Compute and plot the total phase curve
m.load_image('earth.jpg')
m.rotate([0, 1, 0], -180)
total = m.flux(axis=[0, 1, 0], theta=theta)
total /= np.max(total)
ax.plot(theta - 180, total, 'k-', label='Total')

# Compute and plot the total phase curve (numerical)
totalnum = m._flux_numerical(axis=[0, 1, 0], theta=thetanum, tol=1e-5)
totalnum /= np.max(totalnum)
ax.plot(thetanum - 180, totalnum, 'k.')

# Appearance
ax.set_xlim(-180, 180)
ax.set_ylim(-0.05, 1.2)
ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc='best', fontsize=11, ncol=2)
ax.set_xlabel('Sub-observer longitude [deg]', fontsize=24)
ax.set_ylabel('Normalized flux', fontsize=24)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontsize(22)

# Plot the earth images
res = 100
ax_im = [pl.subplot2grid((5, nim), (0, n)) for n in range(nim)]
x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
for n in range(nim):
    i = int(np.linspace(0, npts - 1, nim)[n])
    I = [m.evaluate(axis=[0, 1, 0], theta=theta[i], x=x[j], y=y[j])
         for j in range(res)]
    ax_im[n].imshow(I, origin="lower", interpolation="none", cmap='plasma',
                    extent=(-1, 1, -1, 1))
    ax_im[n].axis('off')
    ax_im[n].set_xlim(-1.05, 1.05)
    ax_im[n].set_ylim(-1.05, 1.05)

# Save
pl.savefig('earthphasecurve.pdf', bbox_inches='tight')
