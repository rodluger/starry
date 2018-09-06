import starry
import starry2
import numpy as np
import matplotlib.pyplot as pl


# starry
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
time = [0.3]
sys.compute(time)

# starry2
star2 = starry2.kepler.Primary()
star2[1] = star[1]
star2[2] = star[2]
b2 = starry2.kepler.Secondary()
b2.L = b.L
b2.prot = b.prot
b2.inc = b.inc
b2.Omega = b.Omega
b2.axis = b.axis
b2[1, 0] = b[1, 0]
sys2 = starry2.kepler.System(star2, b2)
sys2.compute(time, gradient=True)


# Check that the flux agrees
assert np.isclose(b.flux[0], b2.lightcurve[0])

quit()

# Check that the orbital gradients agree
assert np.isclose(b.gradient['time'][0], b2.gradient[0][0])
assert np.isclose(b.gradient["planet1.r"][0], b2.gradient[0][1])
assert np.isclose(b.gradient["planet1.L"][0], b2.gradient[0][2])
assert np.isclose(b.gradient["planet1.prot"][0], b2.gradient[0][3])


print(b.gradient["planet1.tref"][0])

# Check that the map gradients agree
n = 12
for l in range(3):
    for m in range(-l, l + 1):
        grad = b.gradient["planet1.Y_{%d,%d}" % (l, m)]
        grad2 = b2.gradient[0][n]
        assert np.isclose(grad, grad2)
        n += 1
