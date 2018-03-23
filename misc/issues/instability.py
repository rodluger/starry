"""Test various numerical issues."""
import numpy as np
import matplotlib.pyplot as pl
import starry

# Instantiate a system
planet = starry.Planet(4)
star = starry.Star()
system = starry.System([star, planet])

# Zoom in on secondary eclipse ingress
time = np.linspace(0.7114, 0.7122, 10000)
fig, ax = pl.subplots(1, figsize=(8, 7))
for m in range(-4, 5):
    planet.map.reset()
    planet.map[0, 0] = 1
    planet.map[4, m] = 1
    system.compute(time)
    ax.plot(time, m + planet.flux / planet.flux[0], label="m = %d" % m)
pl.legend(ncol=9, fontsize=6)
pl.title("Secondary eclipse ingress for l = 4")
pl.show()
