"""Test the luminosity units."""
import matplotlib.pyplot as pl
import starry
import numpy as np

star = starry.Star()
print(star.map)
print(star.map.evaluate(), star.map.flux())

planet = starry.Planet(L=1e-2, r=0.1, a=3)
sys = starry.System([star, planet])
time = np.linspace(0.25, 1.25, 10000)
sys.compute(time)
pl.plot(time, sys.flux / np.nanmedian(sys.flux), label="Uniform")

star[1] = 0.4
star[2] = 0.26

print(star.map)
print(star.map.evaluate(), star.map.flux())

sys.compute(time)
pl.plot(time, sys.flux / np.nanmedian(sys.flux), label="Limb-Darkened")

pl.legend()
pl.show()
