import numpy as np
from oblate.oblate import sT, dsT

deg = 2
nruns = 1
bo = 0.95
ro = 0.25
f = 0.2
theta = 0.5
eps = 1e-8


Dbo, Dro, Df, Dtheta = dsT(deg, bo, ro, f, theta, nruns)

Dbo_n = (
    sT(deg, bo + eps, ro, f, theta, nruns)
    - sT(deg, bo - eps, ro, f, theta, nruns)
) / (2 * eps)

Dro_n = (
    sT(deg, bo, ro + eps, f, theta, nruns)
    - sT(deg, bo, ro - eps, f, theta, nruns)
) / (2 * eps)

Df_n = (
    sT(deg, bo, ro, f + eps, theta, nruns)
    - sT(deg, bo, ro, f - eps, theta, nruns)
) / (2 * eps)

Dtheta_n = (
    sT(deg, bo, ro, f, theta + eps, nruns)
    - sT(deg, bo, ro, f, theta - eps, nruns)
) / (2 * eps)

# Case 1
for n in [0, 1, 2, 3, 4, 5]:
    print("n = {}".format(n))
    for name, d, d_n in zip(
        ["bo", "ro", "f", "theta"],
        [Dbo, Dro, Df, Dtheta],
        [Dbo_n, Dro_n, Df_n, Dtheta_n],
    ):
        diff = np.abs(d[n] - d_n[n])
        if diff > 10 * eps:
            print(
                "{:7s} {:11.8f}   {:11.8f}   {:4.1e}".format(
                    name, d[n], d_n[n], diff
                )
            )
        else:
            print("{:7s} {:11.8f}   {:11.8f}".format(name, d[n], d_n[n]))
    print("")

