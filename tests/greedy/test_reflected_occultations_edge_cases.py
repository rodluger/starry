import numpy as np
import starry
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pytest


@pytest.mark.parametrize(
    "xs,ys,zs,ro",
    [
        [1.0, 2.0, -1.0, 0.6],
        [1.0, 2.0, -1.0, 5.0],
        [1.0, 2.0, -1.0, 5.0],
        [1.0, 2.0, -1.0, 50.0],
        [0.0, 2.0, -1.0, 0.4],
        [0.0, -1.0, -1.0, 0.4],
        [0.0, -1.0, -1.0, 0.4],
        [1.0, 0.0, -1.0, 0.4],
        [1.0, 0.0, -1.0, 0.1],
        [1.0, 0.0, -1.0, 0.8],
        [1.0, 0.0, 0.0, 0.8],
    ],
)
def test_edges(
    xs, ys, zs, ro, y=[1, 1, 1], ns=100, nb=50, res=999, atol=1e-2, plot=False
):

    # Instantiate
    ydeg = np.sqrt(len(y) + 1) - 1
    map = starry.Map(ydeg=ydeg, reflected=True)
    map[1:, :] = y

    # bo - ro singularities
    singularities = [ro - 1, 0, ro, 1, 1 - ro, 1 + ro]
    labels = [
        "$b_o = r_o - 1$",
        "$b_o = 0$",
        "$b_o = r_o$",
        "$b_o = 1$",
        "$b_o = 1 - r_o$",
        "$b_o = 1 + r_o$",
        "grazing",
        "grazing",
    ]

    # Find where the occultor grazes the terminator
    rs = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    b = -zs / rs
    theta = -np.arctan2(xs, ys)
    tol = 1e-15
    nx = 10
    c = np.cos(theta)
    s = np.sin(theta)
    t = np.tan(theta)
    q2 = c ** 2 + b ** 2 * s ** 2

    # Bottom / top half of occultor
    for sgn0 in [1, -1]:

        # Successively refine x array
        xest = 0
        xdel = ro
        for j in range(10):
            x = np.linspace(xest - xdel, xest + xdel, nx)

            # Divide & conquer
            yomax = 1 + ro
            yomin = -1 - ro
            niter = 0
            xest = 0
            while niter < 100 and np.abs(yomax - yomin) > tol:
                yo_ = 0.5 * (yomax + yomin)
                y = yo_ + sgn0 * np.sqrt(ro ** 2 - x ** 2)
                try:

                    # Scan the x axis for an intersection
                    for i in range(nx):

                        # There are two solutions to the quadratic; pick
                        # the one that's actually on the ellipse
                        p = (x[i] * c - b * s * np.sqrt(q2 - x[i] ** 2)) / q2
                        yt1 = p * s + b * np.sqrt(1 - p ** 2) * c
                        xr = x[i] * c + yt1 * s
                        yr = -x[i] * s + yt1 * c
                        arg1 = np.abs(xr ** 2 + (yr / b) ** 2 - 1)
                        p = (x[i] * c + b * s * np.sqrt(q2 - x[i] ** 2)) / q2
                        yt2 = p * s + b * np.sqrt(1 - p ** 2) * c
                        xr = x[i] * c + yt2 * s
                        yr = -x[i] * s + yt2 * c
                        arg2 = np.abs(xr ** 2 + (yr / b) ** 2 - 1)
                        if arg1 < arg2:
                            if arg1 < 1e-6:
                                yt = yt1
                            else:
                                continue
                        elif arg2 < arg1:
                            if arg2 < 1e-6:
                                yt = yt2
                            else:
                                continue
                        else:
                            continue

                        if (sgn0 == -1) and (y[i] < yt):
                            # Part of the occultor has dipped below the terminator
                            yomin = yo_
                            xest = x[i]
                            raise StopIteration

                        if (sgn0 == 1) and (y[i] > yt):
                            # Part of the occultor has dipped above the terminator
                            yomax = yo_
                            xest = x[i]
                            raise StopIteration

                except StopIteration:
                    niter += 1
                    continue
                else:
                    niter += 1
                    if sgn0 == -1:
                        # The occultor is above the terminator everywhere
                        yomax = yo_
                    else:
                        # The occultor is below the terminator everywhere
                        yomin = yo_

            # Increase res by 10x
            xdel /= 10

        singularities.append(yo_)

    # Arrays over singularities
    yo_s = np.zeros((8, ns))
    logdelta = np.append(-np.inf, np.linspace(-16, -2, ns // 2 - 1))
    delta = np.concatenate((-(10 ** logdelta[::-1]), 10 ** logdelta))
    for i, pt in enumerate(singularities):
        yo_s[i] = pt + delta
    yo_s = yo_s[np.argsort(singularities)]
    labels = list(np.array(labels)[np.argsort(singularities)])

    # Array over full occultation
    yo_full = np.linspace(yo_s[0, 0], yo_s[-1, -1], ns, endpoint=True)

    # All
    yo = np.concatenate((yo_full.reshape(1, -1), yo_s))

    # Compute analytic
    flux = np.zeros_like(yo)
    msg = [["" for n in range(yo.shape[1])] for m in range(yo.shape[0])]
    for i in range(len(yo)):
        for k in tqdm(range(ns)):
            try:
                flux[i, k] = map.flux(
                    xs=xs, ys=ys, zs=zs, xo=0, yo=yo[i, k], ro=ro
                )
            except Exception as e:
                flux[i, k] = 0.0
                msg[i][k] = str(e).split("\n")[0]

    # Compute numerical
    flux_num = np.zeros_like(yo) * np.nan
    flux_num_interp = np.zeros_like(yo) * np.nan
    (lat, lon), (x, y, z) = map.ops.compute_ortho_grid(res)
    img = map.render(xs=xs, ys=ys, zs=zs, res=res).flatten()
    for i in range(len(yo)):
        for k in tqdm(range(ns)):
            idx = x ** 2 + (y - yo[i, k]) ** 2 > ro ** 2
            flux_num_interp[i, k] = np.nansum(img[idx]) * 4 / res ** 2
            if (k == 0) or (k == ns - 1) or (k % (ns // nb) == 0):
                flux_num[i, k] = flux_num_interp[i, k]

        # Adjust the baseline
        offset = np.nanmedian(flux[i]) - np.nanmedian(flux_num_interp[i])
        flux_num_interp[i] += offset
        flux_num[i] += offset

    # Plot
    if plot:

        fig = plt.figure(figsize=(10, 8))
        fig.subplots_adjust(hspace=0.35)
        ax = [
            plt.subplot2grid((40, 40), (0, 0), rowspan=15, colspan=40),
            plt.subplot2grid((40, 40), (20, 0), rowspan=10, colspan=10),
            plt.subplot2grid((40, 40), (20, 10), rowspan=10, colspan=10),
            plt.subplot2grid((40, 40), (20, 20), rowspan=10, colspan=10),
            plt.subplot2grid((40, 40), (20, 30), rowspan=10, colspan=10),
            plt.subplot2grid((40, 40), (30, 0), rowspan=10, colspan=10),
            plt.subplot2grid((40, 40), (30, 10), rowspan=10, colspan=10),
            plt.subplot2grid((40, 40), (30, 20), rowspan=10, colspan=10),
            plt.subplot2grid((40, 40), (30, 30), rowspan=10, colspan=10),
        ]

        # Prepare image for plotting
        img[(img < 0) | (img > 0.0)] = 1
        img = img.reshape(res, res)
        cmap = plt.get_cmap("plasma")
        cmap.set_under("grey")

        # Full light curve
        ax[0].plot(yo[0], flux[0], "k-", lw=1)
        ax[0].plot(yo[0], flux_num[0], "k.", lw=1)
        ax[0].tick_params(labelsize=10)
        ax[0].set_xlabel("$b_o$")
        ax[0].set_ylabel("flux")

        # Each singularity
        for i in range(1, len(yo)):
            ax[0].plot(yo[i], flux[i], lw=3, color="C{}".format(i - 1))
            ax[i].plot(
                2 + logdelta,
                flux[i][: ns // 2],
                lw=2,
                color="C{}".format(i - 1),
            )
            ax[i].plot(
                -(2 + logdelta)[::-1],
                flux[i][ns // 2 :],
                lw=2,
                color="C{}".format(i - 1),
            )
            ax[i].plot(2 + logdelta, flux_num[i][: ns // 2], "k.", ms=2)
            ax[i].plot(
                -(2 + logdelta)[::-1], flux_num[i][ns // 2 :], "k.", ms=2
            )
            ax[i].set_xticks([])
            ax[i].set_yticks([])

            # Show the map
            axins = inset_axes(
                ax[i], width="30%", height="30%", loc=4, borderpad=1
            )
            axins.imshow(
                img,
                origin="lower",
                cmap=cmap,
                extent=(-1, 1, -1, 1),
                vmin=1e-8,
            )
            circ = plt.Circle(
                (0, yo[i, ns // 2]),
                ro,
                fc="k",
                ec="k",
                clip_on=(ro > 0.75),
                zorder=99,
            )
            axins.add_artist(circ)
            axins.annotate(
                labels[i - 1],
                xy=(0.5, -0.1),
                xycoords="axes fraction",
                clip_on=False,
                ha="center",
                va="top",
                fontsize=8,
            )
            axins.set_xlim(-1.01, 1.01)
            axins.set_ylim(-1.01, 1.01)
            axins.axis("off")

        plt.show()

    # Compare
    if not np.allclose(flux, flux_num_interp, atol=atol):
        index = np.unravel_index(
            np.argmax(np.abs(flux - flux_num_interp)), flux.shape
        )
        if index[0] > 0:
            raise ValueError(
                "Error in singular region {}/8: {}".format(
                    index[0], labels[index[0] - 1]
                )
            )
