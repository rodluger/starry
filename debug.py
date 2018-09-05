import starry
import starry2
import numpy as np
import matplotlib.pyplot as pl

star = starry2.kepler.Primary()
star[1] = 0.4
star[2] = 0.26

b = starry2.kepler.Secondary()
b.L = 0.001
b.prot = 1
b.inc = 75
b.Omega = 30
b.axis = [1, 2, 3]
b[1, 0] = 0.5

sys = starry2.kepler.System(star, b)
sys.compute([0.3], gradient=True)
print(b.lightcurve[0])
print(b.gradient[0][14])


#


star = starry.grad.Star()
star[1] = 0.4
star[2] = 0.26

b = starry.grad.Planet()
b.L = 0.001
b.prot = 1
b.inc = 75
b.Omega = 30
b.axis = [1, 2, 3]
b[1, 0] = 0.5

sys = starry.grad.System([star, b])
sys.compute([0.3])
print(b.flux[0])
print(b.gradient['planet1.Y_{1,0}'][0])
