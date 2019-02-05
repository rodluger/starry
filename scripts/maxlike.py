import starry
import numpy as np


map = starry.Map(1)

flux = 1.0 + 0.1 * np.random.randn(10)
xo = np.linspace(-1.5, 1.5, 10)

map.MAP(flux, flux_err=0.1, xo=xo, ro=0.1, L=np.eye(4))
