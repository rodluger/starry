"""Bug in s2 when b + r = 1."""
import matplotlib.pyplot as pl
from starry import Map
import numpy as np
np.random.seed(1234)


# Let's do the l = 3 Earth
m = Map(1)
m[0, 0] = 1
m[1, 0] = 1

# Small occultor
npts = 25
ro = 0.3
eps = 1e-7
xo = np.linspace(0.7 - eps, 0.7 + eps, 1000)
yo = 0
sF = np.array(m.flux(xo=xo, yo=yo, ro=ro))

# Plot
pl.plot(xo, sF)
pl.show()
