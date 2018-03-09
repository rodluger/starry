import starry
import numpy as np

assert starry.__version__ == '0.0.1'

# Check elliptic integrals
assert np.allclose(starry.K(0.5), 1.854074677301372)
assert np.allclose(starry.E(0.5), 1.350643881047675)
assert np.allclose(starry.PI(0.25, 0.5), 2.167619360766556)
