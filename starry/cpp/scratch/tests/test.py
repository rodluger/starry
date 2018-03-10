import starry
import numpy as np

assert starry.__version__ == '0.0.1'

# Check elliptic integrals
assert np.allclose(starry.K(0.5), 1.854074677301372)
assert np.allclose(starry.E(0.5), 1.350643881047675)
assert np.allclose(starry.PI(0.25, 0.5), 2.167619360766556)

# Check factorials
assert np.allclose(starry.factorial(5), 120.)
assert np.allclose(starry.half_factorial(-1), np.sqrt(np.pi))

# Check the Map class
y = starry.Map([0, 1, 0, 0])
assert y.evaluate(0.3, 0.5) == 0.3
