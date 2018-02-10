"""Test the STARRY rotation routines."""
from starry import R
from starry.basis import evaluate_poly, y2p
import numpy as np
import matplotlib.pyplot as pl


# Construct the vector for Y_{1,-1}
y = [0, 1, 0, 0]

# Rotate it
theta = np.pi / 3
u = [1, 0, 0]
y = np.dot(R(2, u, theta), y)

# Convert it to a polynomial basis
p = y2p(y)

# Plot it
fig, ax = pl.subplots(1, figsize=(3, 3))
flux = np.zeros((100, 100)) * np.nan
for i, x in enumerate(np.linspace(-1, 1, 100)):
    for j, y in enumerate(np.linspace(-1, 1, 100)):
        ylim = np.sqrt(1 - x ** 2)
        if y > -ylim and y < ylim:
            flux[j][i] = evaluate_poly(p, x, y)
ax.imshow(flux, cmap='plasma',
          interpolation="none", origin="lower",
          extent=(-1, 1, -1, 1))
ax.axis('off')

pl.show()
