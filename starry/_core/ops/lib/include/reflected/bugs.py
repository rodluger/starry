import numpy as np
import starry

# Things are nan when bo = 0.0 (FIXED?)
ydeg = 1
ops = starry._c_ops.Ops(ydeg, 0, 0, 0)
bo = np.array([0.0])
theta = np.array([0.0])
b = np.array([0.25])
ro = 0.4
bsT = np.ones((1, (ydeg + 1) ** 2))
sT, code, bb, btheta, bbo, bro = ops.sTReflected(b, theta, bo, ro, bsT)
print(sT)
