import starry
import numpy as np
import matplotlib.pyplot as pl

star = starry.Star()
planet = starry.Planet(porb=1, prot=1, a=50)
planet.map[1, 0] = 0.5
system = starry.System([star, planet])
nt = 1000
system.compute(np.linspace(0, 1, nt))

fig, ax = pl.subplots(1, figsize=(8, 3.5))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.plot(planet.x, planet.y, color='k')
asini = 50 * np.sin(planet.inc * np.pi / 180)
acosi = 50 * np.cos(planet.inc * np.pi / 180)
ax.plot(np.array([-asini, asini]), np.array([-acosi, acosi]), 'k--', alpha=0.5)
ax.plot(0, 0, marker="*", color='k', ms=30)
ax.set_xlim(-50, 50)
ax.set_ylim(-40, 40)
ax.axis('off')

res = 100
nim = 15
axw = 0.2
xim, yim = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
inds = np.array(np.linspace(0, nt, nim, endpoint=False), dtype=int)
axis = [0, 1, 0]
theta = np.linspace(0, 360, nim, endpoint=False) + 180
for n in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3]:
    x = 0.5 + (planet.x[inds[n]] / 100)
    y = 0.5 + (planet.y[inds[n]] / 80)
    ax_im = fig.add_axes([x - axw / 2, y - axw / 2, axw, axw])
    I = [planet.map.evaluate(axis=axis, theta=theta[n],
                             x=xim[j], y=yim[j]) for j in range(res)]
    ax_im.imshow(I, origin='lower', extent=(-1, 1, -1, 1), cmap='plasma')
    ax_im.set_xlim(-1.25, 1.25)
    ax_im.set_ylim(-1.25, 1.25)
    ax_im.axis('off')
