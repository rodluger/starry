"""[BROKEN] Numerical error as b --> r."""
import numpy as np
import matplotlib.pyplot as pl
import starry

b = 0.4
ylm = starry.Map(2)
ylm[1, 0] = 1
xo = np.linspace(b - 1e-5, b + 1e-5, 1000)
flux = ylm.flux(xo=xo, yo=0, ro=b)
pl.plot(xo, flux)
pl.plot(b, ylm.flux(xo=b, yo=0, ro=b), 'ro')
pl.show()
