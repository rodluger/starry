"""
A cute animation of the Earth decomposed into successively
higher spherical harmonic degrees.

"""
import starry2
from tqdm import tqdm

# For lmax = 30, this takes about 3 minutes
# since the `load_image` method is NOT OPTIMIZED
lmax = 30
map = starry2.Map(lmax=lmax, nwav=lmax)
for l in tqdm(range(lmax)):
    map.load_image('earth', l, l)
map.show()
