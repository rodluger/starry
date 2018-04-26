"""Smiley spherical harmonic example."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np

# Generate a sample starry map
m = Map(5)
m.set_coeff(5, -3, -2)
m.set_coeff(5, 0, 2)
m.set_coeff(5, 4, 1)

# Render it under consecutive rotations
nax = 8
fig, ax = pl.subplots(1, nax, figsize=(3 * nax, 3))
theta = np.linspace(0, 2 * np.pi, nax, endpoint=False)
x = np.linspace(-1, 1, 300)
y = np.linspace(-1, 1, 300)
x, y = np.meshgrid(x, y)
for i in range(nax):
    I = m.evaluate(axis=[1, 0, 0], theta=theta[i], x=x, y=y)
    ax[i].imshow(I, origin="lower", interpolation="none", cmap='plasma')
    ax[i].axis('off')

# Save
pl.savefig('smiley.pdf', bbox_inches='tight')
