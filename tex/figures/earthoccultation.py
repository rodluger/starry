"""Earth occultation example."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np

# Set up the plot
nim = 12
npts = 100
nptsnum = 10
res = 300
fig = pl.figure(figsize=(12, 5))
ax_im = [pl.subplot2grid((4, nim), (0, n)) for n in range(nim)]
ax_lc = pl.subplot2grid((4, nim), (1, 0), colspan=nim, rowspan=3)

# Instantiate the earth
m = Map(10)
m.load_image('earth')

# Moon params
ro = 0.273
yo = np.linspace(-0.5, 0.5, npts)
yonum = np.linspace(-0.5, 0.5, nptsnum)
xo = np.linspace(-1.5, 1.5, npts)
xonum = np.linspace(-1.5, 1.5, nptsnum)

# Say the occultation occurs over ~1 radian of the Earth's rotation
# That's equal to 24 / (2*pi) hours
time = np.linspace(0, 24 / (2 * np.pi), npts)
timenum = np.linspace(0, 24 / (2 * np.pi), nptsnum)
theta0 = 0
theta = np.linspace(theta0, theta0 + 1., npts, endpoint=True)
thetanum = np.linspace(theta0, theta0 + 1., nptsnum, endpoint=True)

# Compute and plot the flux
F = m.flux(axis=[0, 1, 0], theta=theta, xo=xo, yo=yo, ro=ro)
F /= np.max(F)
ax_lc.plot(time, F, 'k-', label='Total')

# Compute and plot the numerical flux
Fnum = m.flux_numerical(axis=[0, 1, 0], theta=thetanum, xo=xonum,
                        yo=yonum, ro=ro, tol=1e-5)
Fnum /= np.max(Fnum)
ax_lc.plot(timenum, Fnum, 'k.')

# Plot the earth images
x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
for n in range(nim):
    i = int(np.linspace(0, npts - 1, nim)[n])
    I = [m.evaluate(axis=[0, 1, 0], theta=theta[i], x=x[j], y=y[j])
         for j in range(res)]
    ax_im[n].imshow(I, origin="lower", interpolation="none", cmap='plasma',
                    extent=(-1, 1, -1, 1))
    xm = np.linspace(xo[i] - ro + 1e-5, xo[i] + ro - 1e-5, res)
    ax_im[n].fill_between(xm, yo[i] - np.sqrt(ro ** 2 - (xm - xo[i]) ** 2),
                          yo[i] + np.sqrt(ro ** 2 - (xm - xo[i]) ** 2),
                          color='w')
    ax_im[n].axis('off')
    ax_im[n].set_xlim(-1.05, 1.05)
    ax_im[n].set_ylim(-1.05, 1.05)

# Appearance
ax_lc.set_xlabel('Time [hours]', fontsize=24)
ax_lc.set_ylabel('Normalized flux', fontsize=24)
for tick in ax_lc.get_xticklabels() + ax_lc.get_yticklabels():
    tick.set_fontsize(22)

# Save
pl.savefig('earthoccultation.pdf', bbox_inches='tight')
