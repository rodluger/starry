import starry
import starry_beta
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(-5, 10, 1000)
xo = np.linspace(-3, 3, len(theta))
yo = np.zeros_like(xo) + 0.1
ro = 0.1

map = starry.Map(1, 1)
map[1, 0] = 0.5
map.inc = 45
plt.plot(xo, map.flux(theta=theta, xo=xo, yo=yo, ro=ro).eval(), label="dev", lw=3)

map_beta = starry_beta.Map(1)
map_beta[1, 0] = 0.5
map_beta.axis = [0, 1, 1]
plt.plot(xo, map_beta.flux(theta=theta, xo=xo, yo=yo, ro=ro), label="beta", ls="--")

plt.legend()
plt.show()