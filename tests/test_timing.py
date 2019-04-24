"""Test evaluation times compared to the beta version."""
import numpy as np
import starry
import starry_beta
import pytest
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def run():
    """Still working on this one."""
    # Settings
    maxl = 15
    nruns = 10
    yo = 0.3
    
    # Grids
    tgrid = np.array(["ylm", "ld", "both"])
    lgrid = np.arange(1, maxl, dtype=int)
    cgrid = np.array([1, 10, 30, 100, 300, 1000, 3000], dtype=int)
    ggrid = np.array([False, True], dtype=bool)
    rgrid = np.array([0, 0.1])
    v0 = np.empty((tgrid.shape[0], lgrid.shape[0], cgrid.shape[0], 
                   ggrid.shape[0], rgrid.shape[0]))
    v1 = np.empty((tgrid.shape[0], lgrid.shape[0], cgrid.shape[0], 
                   ggrid.shape[0], rgrid.shape[0]))
    
    # Loop over degrees
    for h, maptype in enumerate(tgrid):
        for i, lmax in tqdm(enumerate(lgrid), total=lgrid.shape[0]):

            if maptype == "ylm":
                map_v0 = starry_beta.Map(lmax=lmax)
                map_v0[:, :] = 1
                map_v1 = starry.Map(ydeg=lmax, udeg=0)
                map_v1[:, :] = 1
            elif maptype == "ld":
                map_v0 = starry_beta.Map(lmax=lmax)
                map_v0[:] = 1
                map_v1 = starry.Map(ydeg=0, udeg=lmax)
                map_v1[1:] = 1
            else:
                ydeg = int(lmax // 2 + 1)
                udeg = lmax - ydeg
                map_v0 = starry_beta.Map(lmax=lmax)
                map_v0[:ydeg, :] = 1
                map_v0[:udeg] = 1
                map_v1 = starry.Map(ydeg=ydeg, udeg=udeg)
                map_v1[:, :] = 1
                if udeg > 0:
                    map_v1[1:] = 1

            # Loop over number of cadences
            for j, npts in enumerate(cgrid):
                theta = np.linspace(0, 360, npts)
                xo = np.linspace(-1.1, 1.1, npts)

                # Gradient on / off
                for k, gradient in enumerate(ggrid):

                    # Occultations on / off
                    for l, ro in enumerate(rgrid):
                        tv0 = np.empty(nruns, dtype=float)
                        tv1 = np.empty(nruns, dtype=float)
                        
                        # Run several trials
                        for n in range(nruns):

                            # v0
                            t1 = time.time()
                            res = map_v0.flux(theta=theta, xo=xo, yo=yo, 
                                              ro=ro, gradient=gradient)
                            tv0[n] = time.time() - t1

                            # v1
                            if maptype == "ld":
                                b = np.sqrt(xo ** 2 + yo ** 2)
                                t1 = time.time()
                                res = map_v1.flux(b=b, ro=ro, gradient=gradient)
                                tv1[n] = time.time() - t1
                            else:
                                t1 = time.time()
                                res = map_v1.flux(theta=theta, xo=xo, yo=yo, 
                                                ro=ro, gradient=gradient)
                                tv1[n] = time.time() - t1
                        
                        # Take the median
                        v0[h, i, j, k, l] = np.median(tv0)
                        v1[h, i, j, k, l] = np.median(tv1)

    # Save the results
    np.savez("timing.npz", v0=v0, v1=v1, tgrid=tgrid, lgrid=lgrid, 
             cgrid=cgrid, ggrid=ggrid, rgrid=rgrid)


def plot():
    # Load the data
    data = np.load("timing.npz")
    v0 = data["v0"]
    v1 = data["v1"]
    tgrid = data["tgrid"]
    lgrid = data["lgrid"]
    cgrid = data["cgrid"]
    ggrid = data["ggrid"]
    rgrid = data["rgrid"]

    # Plot
    plt.switch_backend('Qt5Agg')
    fig1, ax1 = plt.subplots(2, 3, figsize=(10, 5))
    fig2, ax2 = plt.subplots(2, 3, figsize=(10, 5))
    for h in range(3):
        for k, gradient in enumerate(ggrid):
            for l, ro in enumerate(rgrid): 
                ax1[k, h].plot(lgrid, v1[h, :, -1, k, l], color="C0",
                                ls="-" if ro else "--",
                                marker='o', ms=3)
                ax1[k, h].plot(lgrid, v0[h, :, -1, k, l], color="C1",
                                ls="-" if ro else "--",
                                marker='o', ms=3)
                ax2[k, h].plot(cgrid, v1[h, -1, :, k, l], color="C0",
                                ls="-" if ro else "--",
                                marker='o', ms=3)
                ax2[k, h].plot(cgrid, v0[h, -1, :, k, l], color="C1",
                                ls="-" if ro else "--",
                                marker='o', ms=3)

    # Appearance
    kwargs = dict(xy=(0, 1), xycoords="axes fraction",
                  ha="left", va="top", xytext=(5, -5), 
                  textcoords="offset points", fontweight="bold")
    
    for ax in (ax1, ax2):
        ax[0, 0].annotate("ylms", **kwargs)
        ax[1, 0].annotate("ylms + grad", **kwargs)
        ax[0, 1].annotate("ld", **kwargs)
        ax[1, 1].annotate("ld + grad", **kwargs)
        ax[0, 2].annotate("both", **kwargs)
        ax[1, 2].annotate("both + grad", **kwargs)
        ax[0, 0].set_ylabel(r"$t\ [\mathrm{sec}]$", fontsize=18)
        ax[1, 0].set_ylabel(r"$t\ [\mathrm{sec}]$", fontsize=18)

        ax[0, 0].plot(np.nan, np.nan, 'k--', label="rot")
        ax[0, 0].plot(np.nan, np.nan, 'k-', label="occ")
        ax[0, 0].plot(np.nan, np.nan, 'C1-', label="v0")
        ax[0, 0].plot(np.nan, np.nan, 'C0-', label="v1")
        ax[0, 0].legend(loc="lower right", fontsize=6)
    
    for axis in ax1.flatten():
        axis.set_yscale("log")

    for axis in ax2.flatten():
        axis.set_yscale("log")
        axis.set_xscale("log")

    for n in range(3):
        ax1[1, n].set_xlabel(r"$l_\mathrm{max}$", fontsize=18)
        ax2[1, n].set_xlabel(r"$n_\mathrm{pts}$", fontsize=18)

    plt.show()

if __name__ == "__main__":
    run()
    plot()