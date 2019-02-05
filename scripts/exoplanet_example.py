import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from starry.ops import LightCurve

# The light curve calculation requires an orbit
orbit = xo.orbits.KeplerianOrbit(period=3.456)

# Params
t = np.linspace(-0.1, 0.1, 500)
lmax = 2
np.random.seed(41)
y = 0.1 * np.random.randn((lmax + 1) ** 2)
y[0] = 1.0

# Instantiate and compute
op = LightCurve(lmax)
light_curve = op.get_light_curve(orbit=orbit, r=0.1, t=t, y=y).eval()

# Plot
plt.plot(t, light_curve, color="C0", lw=2)
plt.ylabel("relative flux")
plt.xlabel("time [days]")
plt.xlim(t.min(), t.max())
plt.show()