import starry
import starry2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


npts = 1000
nsamples = 20
xo = np.linspace(-1.1, 1.1, npts)
lmax_arr = [0, 1, 2, 3, 4, 5, 10, 15]
t = np.zeros((2, 2, len(lmax_arr))) * np.nan

for i, lmax in tqdm(enumerate(lmax_arr), total=len(lmax_arr)):
    for version, map in zip([0, 1], [starry.Map(lmax), starry2.Map(lmax)]):
        map[:,:] = 1
        tk = np.zeros(nsamples) * np.nan
        for gradient in [False, True]:
            for k in range(nsamples):
                tstart = time.time()
                map.flux(xo=xo, yo=0.2, ro=0.1, gradient=gradient)
                tk[k] = time.time() - tstart
            t[version, int(gradient), i] = np.median(tk)


fig, ax = plt.subplots(1)
ax.plot(lmax_arr, t[0, 0], color="C0", ls="-")
ax.plot(lmax_arr, t[0, 0], color="C0", ls="none", marker="o", label="beta")
ax.plot(lmax_arr, t[1, 0], color="C1", ls="-")
ax.plot(lmax_arr, t[1, 0], color="C1", ls="none", marker="o", label="v1.0")
ax.plot(lmax_arr, t[0, 1], color="C0", ls="--")
ax.plot(lmax_arr, t[0, 1], color="C0", ls="none", marker="o", label="beta (+grad)")
ax.plot(lmax_arr, t[1, 1], color="C1", ls="--")
ax.plot(lmax_arr, t[1, 1], color="C1", ls="none", marker="o", label="v1.0 (+grad)")
ax.legend()
ax.set_yscale("log")
plt.show()