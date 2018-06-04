"""Testing systems with inclination."""
import numpy as np
import matplotlib.pyplot as pl
import starry


def R(u, theta):
    """Return the rotation matrix for an angle `theta` and an axis `u`."""
    cost = np.cos(theta)
    sint = np.sin(theta)
    ux, uy, uz = u
    R = np.zeros((3, 3))
    R[0, 0] = cost + ux ** 2 * (1 - cost)
    R[0, 1] = ux * uy * (1 - cost) - uz * sint
    R[0, 2] = ux * uz * (1 - cost) + uy * sint
    R[1, 0] = uy * ux * (1 - cost) + uz * sint
    R[1, 1] = cost + uy ** 2 * (1 - cost)
    R[1, 2] = uy * uz * (1 - cost) - ux * sint
    R[2, 0] = uz * ux * (1 - cost) - uy * sint
    R[2, 1] = uz * uy * (1 - cost) + ux * sint
    R[2, 2] = cost + uz ** 2 * (1 - cost)
    return R


def transform(planet):
    """
    Rotate the planet to the correct orientation.

    This will be coded into `starry` as the default behavior
    when the user specifies the surface map for an inclined planet.
    """
    # Normalize
    planet.axis /= np.sqrt(planet.axis[0] ** 2 +
                           planet.axis[1] ** 2 +
                           planet.axis[2] ** 2)

    inc = planet.inc * np.pi / 180
    Omega = planet.Omega * np.pi / 180

    # Rotate the map to the correct orientation on the sky
    planet.map.rotate(axis=(1, 0, 0), theta=np.pi / 2 - inc)
    planet.map.rotate(axis=(0, 0, 1), theta=Omega)

    # Rotate the axis of rotation in the same way
    planet.axis = np.dot(R((1, 0, 0), np.pi / 2 - inc), planet.axis)
    planet.axis = np.dot(R((0, 0, 1), Omega), planet.axis)


# Instantiate the system
planet = starry.Planet(L=1, porb=1, prot=1)
star = starry.Star()
sys = starry.System([star, planet])
time = np.linspace(0, 1, 1000)
Omega = 0
planet.Omega = Omega

# Set up the figure
fig = pl.figure(figsize=(12, 6))
ax = np.array([[pl.subplot2grid((5, 16), (i, j)) for j in range(8)]
               for i in range(5)])
ax_lc = pl.subplot2grid((5, 16), (0, 9), rowspan=5, colspan=8)
x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
lam = np.linspace(0, 2 * np.pi, 8, endpoint=False)
phase = np.linspace(90, 450, 1000)
phase[np.argmax(phase > 360)] = np.nan
phase[phase > 360] -= 360

# Go through different inclinations
for i, inc in enumerate([90, 60, 45, 30, 0]):
    planet.inc = inc
    planet.map.reset()
    planet.map[1, 0] = -0.5
    planet.axis = (0., 1., 0.)

    # Transform!
    transform(planet)

    # Plot the images
    ax[i, 0].set_ylabel(r'$i=%d^\circ$' % inc)
    for n in range(8):
        img = [planet.map.evaluate(x=x[j], y=y[j], axis=planet.axis,
                                   theta=lam[n] - np.pi / 2)
               for j in range(100)]
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
