"""Stability tests."""
import numpy as np
import matplotlib.pyplot as pl
import starry2
from tqdm import tqdm


def is_even(n):
    """Return true if n is even."""
    if ((n % 2) != 0):
        return False
    else:
        return True


def StarrySExact(barr, r, lmax):
    """Compute s with starry multiprecision."""
    map = starry2.Map(lmax, multi=True)
    map[:, :] = 1
    s = np.zeros(((lmax + 1) ** 2, len(barr)))
    for i in range(len(barr)):
        map.flux(xo=0, yo=barr[i], ro=r)
        s[:, i] = map.s
    return s


def StarryS(barr, r, lmax):
    """Compute s with starry."""
    map = starry2.Map(lmax)
    for ll in range(lmax + 1):
        for mm in range(-ll, ll + 1):
            map[ll, mm] = 1
    s = np.zeros(((lmax + 1) ** 2, len(barr)))
    for i in range(len(barr)):
        map.flux(xo=0, yo=barr[i], ro=r)
        s[:, i] = map.s
    return s


def Compute(r, larr=[0, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20],
            logdelta=-6, logeps=-12, res=50):
    """Run the stability tests."""
    lmax = np.max(larr)
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
    bounds = np.cumsum([0] + [len(bb) for bb in bs]) - 1
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
    ax[1, 0].set_ylabel("Error (scaled)", fontsize=12)
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
    for l in larr:
        for m in range(-l, l + 1):
            err_rel = np.abs(s[n] - s_mp[n])
            err_frac = np.abs((s[n] - s_mp[n]) /
                              max(1e-9, np.max(np.abs(s_mp[n]))))
            if is_even(l - m):
                ax[0, 0].plot(err_rel, color=cmap(l / (lmax + 2)), lw=1)
                ax[1, 0].plot(err_frac, color=cmap(l / (lmax + 2)), lw=1)
            else:
                ax[0, 1].plot(err_rel, color=cmap(l / (lmax + 2)), lw=1)
                ax[1, 1].plot(err_frac, color=cmap(l / (lmax + 2)), lw=1)
            n += 1

    # Dummy curves & a legend
    lines = [None for l in larr]
    labels = ["%d" % l for l in larr]
    for i, l in enumerate(larr):
        lines[i], = ax[0, 0].plot((0, 1), (1e-20, 1e-20),
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


def PaperFigure(larr=[0, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20],
                logdelta=-6, logeps=-12, res=50):
    """Plot the stability figure for the paper."""
    lmax = np.max(larr)
    delta = 10 ** logdelta
    eps = 10 ** logeps
    r = 100
    bs1 = [np.linspace(r - 1, r - 1 + eps, res),
           np.linspace(r - 1 + eps, r - 1 + delta, res),
           np.linspace(r - 1 + delta, r - delta, 3 * res),
           np.linspace(r - delta, r - eps, res),
           np.linspace(r - eps, r + eps, res),
           np.linspace(r + eps, r + delta, res),
           np.linspace(r + delta, r + 1 - delta, 3 * res),
           np.linspace(r + 1 - delta, r + 1 - eps, res),
           np.linspace(r + 1 - eps, r + 1, res)]
    labels1 = [r"$r - 1$", r"$r - 1 + 10^{%d}$" % logeps,
               r"$r - 1 + 10^{%d}$" % logdelta,
               r"$r - 10^{%d}$" % logdelta, r"$r - 10^{%d}$" % logeps,
               r"$r + 10^{%d}$" % logeps, r"$r + 10^{%d}$" % logdelta,
               r"$r + 1 - 10^{%d}$" % logdelta,
               r"$r + 1 - 10^{%d}$" % logeps, r"$r + 1$"]
    b1 = np.concatenate((bs1))
    r = 0.01
    bs0 = [np.linspace(0, eps, res),
           np.linspace(eps, delta, res),
           np.linspace(delta, r - delta, 3 * res),
           np.linspace(r - delta, r - eps, res),
           np.linspace(r - eps, r + eps, res),
           np.linspace(r + eps, r + delta, res),
           np.linspace(r + delta, 1 - r - delta, 3 * res),
           np.linspace(1 - r - delta, 1 - r - eps, res),
           np.linspace(1 - r - eps, 1 - r + eps, res),
           np.linspace(1 - r + eps, 1 - r + delta, res),
           np.linspace(1 - r + delta, 1 - delta, 3 * res),
           np.linspace(1 - delta, 1 - eps, res),
           np.linspace(1 - eps, 1 + eps, res),
           np.linspace(1 + eps, 1 + delta, res),
           np.linspace(1 + delta, r + 1 - delta, 3 * res),
           np.linspace(r + 1 - delta, r + 1 - eps, res),
           np.linspace(r + 1 - eps, r + 1, res)]
    labels0 = [r"$0$",
               r"$10^{%d}$" % logeps,
               r"$10^{%d}$" % logdelta,
               r"$r - 10^{%d}$" % logdelta,
               r"$r - 10^{%d}$" % logeps,
               r"$r + 10^{%d}$" % logeps,
               r"$r + 10^{%d}$" % logdelta,
               r"$1 - r - 10^{%d}$" % logdelta,
               r"$1 - r - 10^{%d}$" % logeps,
               r"$1 - r + 10^{%d}$" % logeps,
               r"$1 - r + 10^{%d}$" % logdelta,
               r"$1 - 10^{%d}$" % logdelta,
               r"$1 - 10^{%d}$" % logeps,
               r"$1 + 10^{%d}$" % logeps,
               r"$1 + 10^{%d}$" % logdelta,
               r"$r + 1 - 10^{%d}$" % logdelta,
               r"$r + 1 - 10^{%d}$" % logeps,
               r"$r + 1$"]
    b0 = np.concatenate((bs0))

    # Set up the figure
    cmap = pl.get_cmap('plasma')
    fig, ax = pl.subplots(2, 2, sharey=True, figsize=(10, 6))
    fig.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.15)
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_ylim(1e-16, 1e0)
    for axis in ax.flatten():
        axis.set_xticks([])
        axis.margins(0, None)
    bounds0 = np.cumsum([0] + [len(bb) for bb in bs0]) - 1
    bounds0[0] = 0
    for v in bounds0:
        for axis in ax[:, 0]:
            axis.axvline(v, lw=0.5, color='k', alpha=0.5, zorder=10, ls='--')
    bounds1 = np.cumsum([0] + [len(bb) for bb in bs1]) - 1
    bounds1[0] = 0
    for v in bounds1:
        for axis in ax[:, 1]:
            axis.axvline(v, lw=0.5, color='k', alpha=0.5, zorder=10, ls='--')
    ax[1, 0].set_xticks(bounds0)
    ax[1, 0].set_xticklabels(labels0, rotation=45, fontsize=7)
    ax[1, 1].set_xticks(bounds1)
    ax[1, 1].set_xticklabels(labels1, rotation=45, fontsize=7)
    for tick in ax[1, 0].xaxis.get_major_ticks() + \
            ax[1, 1].xaxis.get_major_ticks():
        tick.label.set_horizontalalignment('right')
    ax[0, 0].set_title(r"$r = 0.01$", fontsize=14)
    ax[0, 1].set_title(r"$r = 100$", fontsize=14)
    ax[0, 0].set_ylabel("Error (relative)", fontsize=12)
    ax[1, 0].set_ylabel("Error (scaled)", fontsize=12)
    ax[1, 0].set_xlabel("Impact parameter", fontsize=12)
    ax[1, 1].set_xlabel("Impact parameter", fontsize=12)

    # Plot!
    for i, b, r in zip([0, 1], [b0, b1], [0.01, 100]):
        s = StarryS(b, r, lmax)
        s_mp = StarrySExact(b, r, lmax)
        n = 0
        for l in larr:
            for m in range(-l, l + 1):
                err_rel = np.abs(s[n] - s_mp[n])
                err_frac = np.abs((s[n] - s_mp[n]) /
                                  max(1e-9, np.max(np.abs(s_mp[n]))))
                ax[0, i].plot(err_rel, color=cmap(l / (lmax + 2)), lw=1,
                              zorder=-1)
                ax[1, i].plot(err_frac, color=cmap(l / (lmax + 2)), lw=1,
                              zorder=-1)
                n += 1

    # Dummy curves & a legend
    lines = [None for l in larr]
    labels = ["%d" % l for l in larr]
    for i, l in enumerate(larr):
        lines[i], = ax[0, 0].plot((0, 1), (1e-20, 1e-20),
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
        axis.set_rasterization_zorder(0)
    fig.savefig("stability.pdf", bbox_inches='tight', dpi=300)
    pl.close()


if __name__ == "__main__":
    # Compute the ones for the paper
    PaperFigure(logdelta=-3, logeps=-6)

    # Optionally, compute stability tests for a range
    # of occultor sizes. Disabled by default.
    if False:
        pl.rc('text', usetex=False)
        radii = [1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5, 0.75, 1.0,
                 3.0, 5.0, 10.0, 30.0, 50.0, 100.0, 300.0]
        for n, r in tqdm(enumerate(radii), total=len(radii)):
            fig, ax = Compute(r, logdelta=-3, logeps=-6)
            fig.savefig("../stability_test%02d.pdf" % n)
            pl.close()
