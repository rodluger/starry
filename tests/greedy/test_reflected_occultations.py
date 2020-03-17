import numpy as np
import starry
import matplotlib.pyplot as plt
from datetime import datetime
import pytest


@pytest.mark.parametrize(
    "xs,ys,zs",
    [
        [0, 1, 1],
        [-1, 0, 1],
        [0.5, 1, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0, 0, 1],
    ],
)
def test_X(xs, ys, zs, theta=0, ro=0.1, res=300, ydeg=2, tol=1e-3, plot=True):
    # Params
    npts = 250
    xo = np.linspace(-1.5, 1.5, npts)
    yo = np.linspace(-0.3, 0.5, npts)
    theta = 0
    ro = 0.1
    res = 300
    ydeg = 2
    tol = 1e-3

    # Instantiate
    map = starry.Map(ydeg=ydeg, reflected=True)

    # Analytic
    X = map.design_matrix(
        xs=xs, ys=ys, zs=zs, theta=theta, xo=xo, yo=yo, ro=ro
    )

    # Numerical
    x, y, z = map.ops.compute_ortho_grid(res)
    image = np.zeros((map.Ny, res * res))
    image[0] = map.render(theta=theta, xs=xs, ys=ys, zs=zs, res=res).flatten()
    n = 1
    for l in range(1, map.ydeg + 1):
        for m in range(-l, l + 1):
            map.reset()
            map[l, m] = 1
            image[n] = (
                map.render(theta=theta, xs=xs, ys=ys, zs=zs, res=res).flatten()
            ) - image[0]
            n += 1
    X_num = np.zeros_like(X)
    for k in range(len(xo)):
        idx = (x - xo[k]) ** 2 + (y - yo[k]) ** 2 > ro ** 2
        for n in range(map.Ny):
            X_num[k, n] = np.nansum(image[n][idx])
    X_num *= 4 / res ** 2

    # Plot
    if plot:

        fig, ax = plt.subplots(
            ydeg + 1, 2 * ydeg + 1, figsize=(9, 6), sharex=True, sharey=True
        )
        for axis in ax.flatten():
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.spines["left"].set_visible(False)
        n = 0
        for i, l in enumerate(range(ydeg + 1)):
            for j, m in enumerate(range(-l, l + 1)):
                j += ydeg - l
                med = np.median(X_num[:, n])
                ax[i, j].plot(X[:, n] - med, lw=2)
                ax[i, j].plot(X_num[:, n] - med, lw=1)
                n += 1

        fig.savefig(
            "test_X_{}.pdf".format(datetime.now().strftime("%d%m%Y%H%M%S")),
            bbox_inches="tight",
        )
        plt.close()

    # Compare
    diff = (X - X_num).flatten()
    assert np.max(np.abs(diff)) < tol
