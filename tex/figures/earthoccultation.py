"""Earth occultation example."""
from starry import starry
import matplotlib.pyplot as pl
import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)


# Set up the plot
nim = 12
npts = 100
res = 300
fig = pl.figure(figsize=(12, 5))
ax_im = [pl.subplot2grid((4, nim), (0, n)) for n in range(nim)]
ax_lc = pl.subplot2grid((4, nim), (1, 0), colspan=nim, rowspan=3)

# Instantiate the earth
y = starry(10, 'earth')

# Moon params
r = 0.273
y0 = np.linspace(-0.5, 0.5, npts)
x0 = np.linspace(-1.5, 1.5, npts)

# Say the occultation occurs over ~1 radian of the Earth's rotation
# That's equal to 24 / (2*pi) hours
time = np.linspace(0, 24 / (2 * np.pi), npts)
theta0 = 0
theta = np.linspace(theta0, theta0 + 1., npts, endpoint=True)

# Compute and plot the flux
F = y.flux(u=[0, 1, 0], theta=theta, x0=x0, y0=y0, r=r)
F /= np.max(F)
ax_lc.plot(time, F, 'k-', label='Total')

# Plot the earth images
for n in range(nim):
    i = int(np.linspace(0, npts - 1, nim)[n])
    I = y.render(res=res, u=[0, 1, 0], theta=theta[i], x0=x0[i], y0=y0[i], r=r)
    ax_im[n].imshow(I, origin="lower", interpolation="none", cmap='plasma')
    ax_im[n].axis('off')

# Appearance
ax_lc.set_xlabel('Time [hours]', fontsize=24)
ax_lc.set_ylabel('Normalized flux', fontsize=24)
for tick in ax_lc.get_xticklabels() + ax_lc.get_yticklabels():
    tick.set_fontsize(22)

# Save
pl.savefig('earthoccultation.pdf', bbox_inches='tight')
