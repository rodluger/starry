from starry import R, A, S
import numpy as np
import matplotlib.pyplot as pl
from starry.integrals import slm


# Occultation parameters
r = 0.3
b = np.linspace(0, 2, 500)

# Construct a vector of Ylm coefficients
lmax = 3
y = np.zeros((lmax + 1) ** 2, dtype=float)
y[8] = 1

# Rotate it (null rotation for now)
RRy = y

# Convert to the Greens basis
ARRy = np.dot(A(lmax), RRy)


# DEBUG
# Let's compute the light curves for each of the Greens basis terms
for l in range(lmax):
    for m in range(-l, l + 1):
        F = np.array([slm(l, m, b[i], r) for i, _ in enumerate(b)])
        pl.plot(b, F - np.nanmean(F))
pl.show()
quit()


# Compute the integrals
ST = np.array([S(lmax, b[i], r).T for i, _ in enumerate(b)])

# Magic!
F = np.dot(ST, ARRy)

# Plot
pl.plot(b, F)
pl.show()
