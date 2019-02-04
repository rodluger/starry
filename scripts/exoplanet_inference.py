import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from starry2.ops import LightCurve

# The light curve calculation requires an orbit
orbit = xo.orbits.KeplerianOrbit(period=3.456)

# Params
t = np.linspace(-0.1, 0.1, 100)
lmax = 2
u = [0.3, 0.2]
y = np.zeros(9)
y[0] = 1

# Instantiate and compute
op = LightCurve(lmax)
light_curve = op.get_light_curve(orbit=orbit, r=0.1, t=t, y=y, u=u).eval()

# Plot
plt.plot(t, light_curve, color="C0", lw=2)
plt.ylabel("relative flux")
plt.xlabel("time [days]")
plt.xlim(t.min(), t.max())
plt.show()