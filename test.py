from starry import R, A, S
from starry.integrals import brute
import numpy as np
import matplotlib.pyplot as pl

# Occultation parameters
r = 0.3
b = np.linspace(0, 2, 500)
bbrute = np.linspace(0, 2, 30)

# Construct a vector of Ylm coefficients
lmax = 3
y = np.zeros((lmax + 1) ** 2, dtype=float)
y[0] = 1

# Rotate it (null rotation for now)
RRy = y

# Convert to the Greens basis
ARRy = np.dot(A(lmax), RRy)

# Compute the integrals
ST = np.array([S(lmax, b[i], r).T for i, _ in enumerate(b)])

# Magic!
F = np.dot(ST, ARRy)

# Compute the brute flux
FBrute = [brute(y, 0, bbrute[i], r, res=300) for i, _ in enumerate(bbrute)]

# Plot
pl.plot(b, F)
pl.plot(bbrute, FBrute, '.')
pl.show()
