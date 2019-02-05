import starry
import numpy as np
import matplotlib.pyplot as plt

lmax = 2
nt = 100
map = starry.Map(lmax)
np.random.seed(43)
map[:, :] = np.random.randn(map.N)


flux = 1.0 + 0.1 * np.random.randn(nt)
xo = np.linspace(-1.5, 1.5, nt)
yo = np.linspace(-0.1, 0.3, nt)

A = map.MAP(flux, C=0.1, xo=xo, yo=yo, ro=0.1, zo=1)

plt.plot(map.flux(xo=xo, yo=yo, ro=0.1, zo=1))

plt.plot(np.dot(A, map.y))

plt.show()