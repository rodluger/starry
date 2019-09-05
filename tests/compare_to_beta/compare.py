"""
Functions to compare stuff to the beta version. A work in progress.

"""
import starry
import starry_beta
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import time
from tqdm import tqdm

starry.config.lazy = False


def compare_plot():

    theta = np.linspace(-5, 10, 1000)
    xo = np.linspace(-3, 3, len(theta))
    yo = np.zeros_like(xo) + 0.1
    ro = 0.1

    map = starry.Map(ydeg=1, udeg=1)
    map[1, 0] = 0.5
    map.inc = 45
    plt.plot(xo, map.flux(theta=theta, xo=xo, yo=yo, ro=ro), label="v1", lw=3)

    map_beta = starry_beta.Map(1)
    map_beta[1, 0] = 0.5
    map_beta.axis = [0, 1, 1]
    plt.plot(
        xo,
        map_beta.flux(theta=theta, xo=xo, yo=yo, ro=ro),
        label="beta",
        ls="--",
    )

    plt.legend()
    plt.show()


def time_flux(ydeg, occultation=False, npts=np.logspace(0, 4, 10), ntimes=100):

    # Define the new starry function
    map = starry.Map(ydeg=ydeg)
    map[1:, :] = 1
    map.inc = 45
    t_flux = lambda theta, xo, yo, ro: map.flux(
        theta=theta, xo=xo, yo=yo, ro=ro
    )

    # Define the starry beta function
    map_beta = starry_beta.Map(ydeg)
    map_beta[1:, :] = 1
    map_beta.axis = [0, 1, 1]
    b_flux = lambda theta, xo, yo, ro: map_beta.flux(
        theta=theta, xo=xo, yo=yo, ro=ro
    )

    if occultation:
        ro = 0.1
    else:
        ro = 0.0

    t_time = np.zeros_like(npts)
    b_time = np.zeros_like(npts)
    for i in tqdm(range(len(npts))):

        theta = np.linspace(-180, 180, int(npts[i]))
        xo = np.linspace(-1.0, 1.0, int(npts[i]))
        yo = np.zeros_like(xo) + 0.1

        for t, flux in zip([t_time, b_time], [t_flux, b_flux]):
            elapsed = np.zeros(ntimes)
            for k in range(ntimes):
                tstart = time.time()
                flux(theta, xo, yo, ro)
                elapsed[k] = time.time() - tstart
            t[i] = np.median(elapsed)

    return b_time, t_time


def compare_times():

    ydeg = [1, 2, 3, 5, 10, 15]
    npts = np.logspace(0, 4, 10)

    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 10))
    ax = ax.flatten()

    for i in range(len(ydeg)):
        for occultation, fillstyle, ls in zip(
            [False, True], ["none", "full"], ["--", "-"]
        ):
            b_time, t_time = time_flux(
                ydeg[i], npts=npts, occultation=occultation
            )
            ax[i].plot(
                npts, t_time, "C0o", fillstyle=fillstyle, ls="none", ms=3
            )
            ax[i].plot(npts, t_time, "C0", ls=ls, lw=1, alpha=0.5)
            ax[i].plot(
                npts, b_time, "C1o", fillstyle=fillstyle, ls="none", ms=3
            )
            ax[i].plot(npts, b_time, "C1", ls=ls, lw=1, alpha=0.5)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].annotate(
                r"$\mathcal{l} = %s$" % ydeg[i],
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=12,
            )

    ax[0].plot([], [], "C0-", label="v1")
    ax[0].plot([], [], "C1-", label="beta")
    ax[0].plot([], [], "k--", label="rotation")
    ax[0].plot([], [], "k-", label="occultation")
    ax[0].legend(fontsize=8, loc="upper right")
    for i in [4, 5]:
        ax[i].set_xlabel("Number of points", fontsize=14)
    for i in [0, 2, 4]:
        ax[i].set_ylabel("Time [seconds]", fontsize=14)
    plt.show()


compare_times()
