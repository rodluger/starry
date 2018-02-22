"""Earth spherical harmonic example."""
from starry import starry
import matplotlib.pyplot as pl
import numpy as np

# Generate a sample starry map
y = starry(10, 'earth')

# Start centered at longitude 180 W
y.rotate([0, 1, 0], -np.pi)

# Render it under consecutive rotations
nax = 8
fig, ax = pl.subplots(1, nax, figsize=(3 * nax, 3))
theta = np.linspace(0, 2 * np.pi, nax, endpoint=False)
for i in range(nax):
    I = y.render(res=300, u=[0, 1, 0], theta=-theta[i])
    ax[i].imshow(I, origin="lower", interpolation="none", cmap='plasma')
    ax[i].axis('off')

# Save
pl.savefig('earth.pdf', bbox_inches='tight')
