"""Stability tests for large occultors."""
import numpy as np
import matplotlib.pyplot as pl
import starry
from tqdm import tqdm


def is_even(n):
    """Return true if n is even."""
    if ((n % 2) != 0):
        return False
    else:
        return True


def StarrySExact(barr, r, lmax):
    """Compute s with starry multiprecision."""
    map = starry.multi.Map(lmax)
    map[:] = 1
    s = np.zeros(((lmax + 1) ** 2, len(barr)))
    for i in range(len(barr)):
        map.flux(xo=0, yo=barr[i], ro=r)
        s[:, i] = map.s
    return s


def StarryS(barr, r, lmax):
    """Compute s with starry."""
    map = starry.Map(lmax)
    for ll in range(lmax + 1):
        for mm in range(-ll, ll + 1):
            map[ll, mm] = 1
    s = np.zeros(((lmax + 1) ** 2, len(barr)))
    for i in range(len(barr)):
        map.flux(xo=0, yo=barr[i], ro=r)
        s[:, i] = map.s
    return s


def Compute(r, lmax=8, logdelta=-6, logeps=-12, res=50):
    """Run the stability tests."""
    delta = 10 ** logdelta
    eps = 10 ** logeps
    if r > 1:
        bs = [np.linspace(r - 1, r - 1 + eps, res),
              np.linspace(r - 1 + eps, r - 1 + delta, res),
              np.linspace(r - 1 + delta, r - delta, 3 * res),
              np.linspace(r - delta, r - eps, res),
              np.linspace(r - eps, r + eps, res),
              np.linspace(r + eps, r + delta, res),
              np.linspace(r + delta, r + 1 - delta, 3 * res),
              np.linspace(r + 1 - delta, r + 1 - eps, res),
              np.linspace(r + 1 - eps, r + 1, res)]
        labels = [r"$r - 1$", r"$r - 1 + 10^{%d}$" % logeps,
                  r"$r - 1 + 10^{%d}$" % logdelta,
                  r"$r - 10^{%d}$" % logdelta, r"$r - 10^{%d}$" % logeps,
                  r"$r + 10^{%d}$" % logeps, r"$r + 10^{%d}$" % logdelta,
                  r"$r + 1 - 10^{%d}$" % logdelta,
                  r"$r + 1 - 10^{%d}$" % logeps, r"$r + 1$"]
    else:
        bs = [np.linspace(r - 1, r - 1 + eps, res),
              np.linspace(r - 1 + eps, r - 1 + delta, res),
              np.linspace(r - 1 + delta, 0 - delta, 3 * res),
              np.linspace(0 - delta, 0 - eps, res),
              np.linspace(0 - eps, 0, res),
              np.linspace(0, r - eps, res),
              np.linspace(r - eps, r + eps, res),
              np.linspace(r + eps, r + delta, res),
              np.linspace(r + delta, r + 1 - delta, 3 * res),
              np.linspace(r + 1 - delta, r + 1 - eps, res),
              np.linspace(r + 1 - eps, r + 1, res)]
        labels = [r"$r - 1$", r"$r - 1 + 10^{%d}$" % logeps,
                  r"$r - 1 + 10^{%d}$" % logdelta,
                  r"$-10^{%d}$" % logdelta, r"$-10^{%d}$" % logeps, r"$0$",
                  r"$r - 10^{%d}$" % logeps,
                  r"$r + 10^{%d}$" % logeps, r"$r + 10^{%d}$" % logdelta,
                  r"$r + 1 - 10^{%d}$" % logdelta,
                  r"$r + 1 - 10^{%d}$" % logeps, r"$r + 1$"]
    b = np.concatenate((bs))

    # Set up the figure
    cmap = pl.get_cmap('plasma')
    fig, ax = pl.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 6))
    fig.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.15)
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylim(1e-16, 1e0)
    for axis in ax.flatten():
        axis.set_xticks([])
        axis.margins(0, None)
    bounds = np.cumsum([0] + [len(b) for b in bs]) - 1
    bounds[0] = 0
    for v in bounds:
        for axis in ax.flatten():
            axis.axvline(v, lw=0.5, color='k', alpha=0.5, zorder=10, ls='--')
    ax[1, 0].set_xticks(bounds)
    ax[1, 0].set_xticklabels(labels, rotation=45, fontsize=10)
    ax[1, 1].set_xticks(bounds)
    ax[1, 1].set_xticklabels(labels, rotation=45, fontsize=10)
    for tick in ax[1, 0].xaxis.get_major_ticks() + \
            ax[1, 1].xaxis.get_major_ticks():
        tick.label.set_horizontalalignment('right')
    ax[0, 0].set_title("Even terms", fontsize=14)
    ax[0, 1].set_title("Odd terms", fontsize=14)
    ax[0, 0].set_ylabel("Error (relative)", fontsize=12)
    ax[1, 0].set_ylabel("Error (fractional)", fontsize=12)
    ax[1, 0].set_xlabel("Impact parameter", fontsize=12)
    ax[1, 1].set_xlabel("Impact parameter", fontsize=12)
    if r < 1:
        fig.suptitle("r = %.5f" % r, fontweight='bold', fontsize=14)
    else:
        fig.suptitle("r = %.1f" % r, fontweight='bold', fontsize=14)

    # Plot!
    s = StarryS(b, r, lmax)
    s_mp = StarrySExact(b, r, lmax)
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            err_rel = np.abs(s[n] - s_mp[n])
            err_frac = np.abs((s[n] - s_mp[n]) / s_mp[n])
            if is_even(l - m):
                ax[0, 0].plot(err_rel, color=cmap(l / (lmax + 2)), lw=1)
                ax[1, 0].plot(err_frac, color=cmap(l / (lmax + 2)), lw=1)
            else:
                ax[0, 1].plot(err_rel, color=cmap(l / (lmax + 2)), lw=1)
                ax[1, 1].plot(err_frac, color=cmap(l / (lmax + 2)), lw=1)
            n += 1

    # Dummy curves & a legend
    lines = [None for l in range(lmax + 1)]
    labels = ["%d" % l for l in range(lmax + 1)]
    for l in range(lmax + 1):
        lines[l], = ax[0, 0].plot((0, 1), (1e-20, 1e-20),
                                  color=cmap(l / (lmax + 2)), lw=2)
    leg = fig.legend(lines, labels, (0.925, 0.35), title='Degree')
    leg.get_title().set_fontweight('bold')

    for axis in ax.flatten():
        axis.axhline(1e-3, ls='--', lw=1, color='k', alpha=0.5)
        axis.axhline(1e-6, ls='--', lw=1, color='k', alpha=0.5)
        axis.axhline(1e-9, ls='--', lw=1, color='k', alpha=0.5)
        axis.annotate("ppt", xy=(5, 1e-3), xycoords="data", xytext=(3, -3),
                      textcoords="offset points", ha="left", va="top",
                      alpha=0.75)
        axis.annotate("ppm", xy=(5, 1e-6), xycoords="data", xytext=(3, -3),
                      textcoords="offset points", ha="left", va="top",
                      alpha=0.75)
        axis.annotate("ppb", xy=(5, 1e-9), xycoords="data", xytext=(3, -3),
                      textcoords="offset points", ha="left", va="top",
                      alpha=0.75)
    return fig, ax


if __name__ == "__main__":
    '''
    # Compute the ones for the paper
    fig, ax = Compute(0.01, logdelta=-3, logeps=-6)
    fig.savefig("stability.pdf", bbox_inches='tight')
    pl.close()
    fig, ax = Compute(100, logdelta=-3, logeps=-6)
    fig.savefig("stability_eclipse.pdf", bbox_inches='tight')
    pl.close()
    '''

    # Now compute the rest, but first
    # disable LaTeX to speed up the plotting
    pl.rc('text', usetex=False)
    radii = [1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5, 0.75, 1.0,
             3.0, 5.0, 10.0, 30.0, 50.0, 100.0, 300.0]
    for n, r in tqdm(enumerate(radii), total=len(radii)):
        fig, ax = Compute(r, logdelta=-3, logeps=-6)
        fig.savefig("../stability_test%02d.pdf" % n)
        pl.close()
