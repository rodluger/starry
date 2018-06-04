"""Earth spherical harmonic example."""
from starry import Map
import matplotlib.pyplot as pl
import numpy as np

# Generate a sample starry map
m = Map(10)
m.load_image('earth')

# Start centered at longitude 180 W
m.rotate([0, 1, 0], -180)

# Render it under consecutive rotations
nax = 8
res = 300
fig, ax = pl.subplots(1, nax, figsize=(3 * nax, 3))
theta = np.linspace(0, 360, nax, endpoint=False)
x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
for i in range(nax):
    # starry functions accept vector arguments, but not matrix arguments,
    # so we need to iterate below:
    I = [m.evaluate(axis=[0, 1, 0], theta=-theta[i], x=x[j], y=y[j]) for j in range(res)]
    ax[i].imshow(I, origin="lower", interpolation="none", cmap='plasma')
    ax[i].axis('off')

# Save
pl.savefig('earth.pdf', bbox_inches='tight')
