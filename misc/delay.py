"""Testing light travel time delay."""
import numpy as np
import matplotlib.pyplot as pl
import starry

star = starry.Star()
planet = starry.Planet(L=1e-1, prot=1, porb=1, r=1, Omega=1)
sys = starry.System([star, planet])
planet.a /= 50
time = np.linspace(0.4999, 0.5001, 100000)

for R in [0, 1, 5, 10]:
    sys.scale = R
    sys.compute(time)
    a = planet.a * R * 0.00464913034
    pl.plot(time, planet.flux, label="a=%.3f AU" % a)

    print("Secondary eclipse time: ", time[np.argmin(planet.flux)])
    print("Analytic solution: ", 0.5 + 2 * planet.a * R * 6.957e8 /
          299792458 / 86400)

pl.legend()
pl.show()
