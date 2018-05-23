"""Testing light travel time delay."""
import numpy as np
import matplotlib.pyplot as pl
import starry

star = starry.Star()
planet = starry.Planet(L=1e-1, prot=1, porb=1, r=1)
planet.map[1, 0] = 1 / np.sqrt(3)
sys = starry.System([star, planet])

time = np.linspace(-0.1, 1.1, 10000)

for R in [0, 1, 5, 10]:
    sys.scale = R
    sys.compute(time)
    a = planet.a * R * 0.00464913034
    pl.plot(time, sys.flux, label="a=%.1f AU" % a)

pl.legend()
pl.show()
