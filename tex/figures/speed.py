"""Starry speed tests."""
import starry
import timeit
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import matplotlib.pyplot as pl
from tqdm import tqdm
import numpy as np


def speed():
    """Run the main speed tests for phase curves and occultations."""
    # Constants
    number = 3
    lmax = 5
    nN = 8
    Nmax = 5
    time_phase = np.zeros((lmax + 1, 2 * lmax + 1, nN))
    time_occ = np.zeros((lmax + 1, 2 * lmax + 1, nN))
    Narr = np.logspace(1, Nmax, nN)

    # Loop over number of cadences
    for i, N in tqdm(enumerate(Narr), total=nN):

        # Compute for each Ylm
        theta = np.linspace(0, 2 * np.pi, N)
        xo = np.linspace(-1., 1., N)
        for l in range(lmax + 1):
            ylm = starry.Map(l)
            for m in range(-l, l + 1):
                ylm.reset()
                ylm[l, m] = 1
                builtins.__dict__.update(locals())

                # Phase curve
                time_phase[l, m, i] = timeit.timeit(
                    'ylm.flux(u=[0,1,0], theta=theta)',
                    number=number) / number

                # Occultation (no rotation)
                time_occ[l, m, i] = timeit.timeit(
                    'ylm.flux(xo=xo, yo=0.5, ro=0.1)',
                    number=number) / number

    # Plot
    fig, ax = pl.subplots(2, 2, figsize=(7, 5))
    ax = ax.flatten()
    for l in range(lmax + 1):
        time_phasel = np.zeros(nN)
        time_occl = np.zeros(nN)
        for m in range(-l, l + 1):
            time_phasel += time_phase[l, m]
            time_occl += time_occ[l, m]
        time_phasel /= (2 * l + 1)
        time_occl /= (2 * l + 1)
        color = 'C%d' % l
        ax[0].plot(Narr, time_phasel, 'o', ms=2, color=color)
        ax[0].plot(Narr, time_phasel, '-',
                   lw=0.5, color=color, label=r"$l = %d$" % l)
        ax[1].plot(Narr, time_occl, 'o', ms=2, color=color)
        ax[1].plot(Narr, time_occl, '-',
                   lw=0.5, color=color, label=r"$l = %d$" % l)

    l = 5
    for m in range(l + 1):
        color = 'C%d' % m
        ax[2].plot(Narr, time_phase[l, m], 'o', ms=2, color=color)
        ax[2].plot(Narr, time_phase[l, m], '-',
                   lw=0.5, color=color, label=r"$m = %d$" % m)
        ax[3].plot(Narr, time_occ[l, m], 'o', ms=2, color=color)
        ax[3].plot(Narr, time_occ[l, m], '-',
                   lw=0.5, color=color, label=r"$m = %d$" % m)

    # Tweak and save
    ax[0].legend(fontsize=7, loc='upper left')
    ax[1].legend(fontsize=7, loc='upper left')
    ax[2].legend(fontsize=7, loc='upper left')
    ax[3].legend(fontsize=7, loc='upper left')
    ax[0].set_ylabel("Time [s]", fontsize=10)
    ax[2].set_ylabel("Time [s]", fontsize=10)
    ax[2].set_xlabel("Number of points", fontsize=10)
    ax[3].set_xlabel("Number of points", fontsize=10)
    ax[0].set_title('Phase curves')
    ax[1].set_title('Occultations')
    for axis in ax:
        axis.grid(True)
        for line in axis.get_xgridlines() + axis.get_ygridlines():
            line.set_linewidth(0.2)
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_ylim(1e-5, 1e0)
    fig.savefig("speed.pdf", bbox_inches='tight')


if __name__ == "__main__":
    speed()
