"""Testing systems with inclination."""
import numpy as np
import matplotlib.pyplot as pl
import starry


def transform(planet):
    """
    Rotate the planet to the correct orientation.

    This will be coded into `starry` as the default behavior
    when the user specifies the surface map for an inclined planet.
    """
    # This is the axis perpendicular to the orbital plane
    planet.axis = (np.sin(planet.Omega * np.pi / 180),
                   np.sin(planet.inc * np.pi / 180) *
                   np.cos(planet.Omega * np.pi / 180),
                   np.cos(planet.inc * np.pi / 180))

    # The dot product of `yhat` with `axis` is the cosine
    # of the angle `theta` we need to rotate the map by to get
    # it in the right frame
    costheta = planet.axis[1]
    theta = np.arccos(costheta)

    # The cross product of `yhat` with `planet.axis` gives the axis
    # of rotation `axis_prime` for this transformation
    axis_prime = [planet.axis[2], 0, -planet.axis[0]]
    axis_prime /= np.sqrt(planet.axis[2] ** 2 + planet.axis[0] ** 2)

    # Perform the transformation
    planet.map.rotate(axis=axis_prime, theta=theta)


# Instantiate the system
planet = starry.Planet(L=1, porb=1, prot=1)
star = starry.Star()
sys = starry.System([star, planet])
time = np.linspace(0, 1, 1000)
Omega = 45
planet.Omega = Omega

# Set up the figure
fig = pl.figure(figsize=(12, 6))
ax = np.array([[pl.subplot2grid((5, 16), (i, j)) for j in range(8)]
               for i in range(5)])
ax_lc = pl.subplot2grid((5, 16), (0, 9), rowspan=5, colspan=8)
x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
lam = np.linspace(0, 2 * np.pi, 8, endpoint=False)
phase = np.linspace(90, 450, 1000)
phase[phase > 360] -= 360

# Go through different inclinations
for i, inc in enumerate([90, 60, 45, 30, 0]):
    planet.inc = inc
    planet.map.reset()
    planet.map[1, 0] = -0.5
    transform(planet)

    # Plot the images
    ax[i, 0].set_ylabel(r'$i=%d^\circ$' % inc)
    for n in range(8):
        img = [planet.map.evaluate(x=x[j], y=y[j], axis=planet.axis,
                                   theta=lam[n] - np.pi / 2) for j in range(100)]
        ax[i, n].imshow(img, origin="lower", interpolation="none",
                        cmap="plasma", extent=(-1, 1, -1, 1))
        ax[i, n].contour(img, origin="lower", extent=(-1, 1, -1, 1),
                         colors='k', linewidths=1)
        ax[i, n].set_frame_on(False)
        ax[i, n].set_xticks([])
        ax[i, n].set_yticks([])

    # Compute and plot the phase curve
    sys.compute(time)
    ax_lc.plot(phase, planet.flux, label=r"$i=%d^\circ$" % inc)

# Appearance
for n in range(8):
    ax[-1, n].set_xlabel(r'$\lambda = %d^\circ$' % (lam[n] * 180 / np.pi))
ax_lc.legend()
ax_lc.set_xlabel(r'$\lambda\, (^\circ)$')
ax_lc.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
ax_lc.set_ylabel('Flux')
pl.show()
