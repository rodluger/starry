"""Smiley spherical harmonic example."""
from starry import starry
import matplotlib.pyplot as pl
import numpy as np

# Generate a sample starry map
y = starry(5)
y[5, -3] = -2
y[5, 0] = 2
y[5, 4] = 1

# Render it under consecutive rotations
nax = 8
fig, ax = pl.subplots(1, nax, figsize=(3 * nax, 3))
theta = np.linspace(0, 2 * np.pi, nax, endpoint=False)
for i in range(nax):
    I = y.render(res=300, u=[1, 0, 0], theta=theta[i])
    ax[i].imshow(I, origin="lower", interpolation="none", cmap='plasma')
    ax[i].axis('off')

# Save
pl.savefig('smiley.pdf', bbox_inches='tight')
