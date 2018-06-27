"""Stability tests for the gradients."""
import numpy as np
import matplotlib.pyplot as pl
import starry


def is_even(n):
    """Return true if n is even."""
    if ((n % 2) != 0):
        return False
    else:
        return True


def StarryDExact(barr, r, lmax, d='b', tiny=1e-8):
    """Compute dF/d{b,r} with starry multiprecision."""
    map = starry.multi.Map(lmax)
    res = np.zeros((lmax + 1, len(barr)))
    for ll in range(lmax + 1):
        map.reset()
        for mm in range(-ll, ll + 1):
            map[ll, mm] = 1
        if d == 'b':
            res[ll] = map._dfluxdyo(xo=0, yo=barr, ro=r)
        elif d == 'r':
            res[ll] = map._dfluxdro(xo=0, yo=barr, ro=r)
        else:
            raise ValueError("Invalid derivative name.")
    return res


def _StarryDExact(barr, r, lmax, d='b', tiny=1e-16):
    """Compute dF/d{b,r} for each degree with starry.grad."""
    map = starry.grad.Map(lmax)
    res = np.zeros((lmax + 1, len(barr)))
    for ll in range(lmax + 1):
        map.reset()
        for mm in range(-ll, ll + 1):
            map[ll, mm] = 1
        if d == 'b':
            map.flux(xo=0, yo=barr - barr * tiny, ro=r)
            f1 = map.gradient['yo']
            map.flux(xo=0, yo=barr + barr * tiny, ro=r)
            f2 = map.gradient['yo']
            res[ll] = 0.5 * (f1 + f2)
        elif d == 'r':
            map.flux(xo=0, yo=barr, ro=r - r * tiny)
            f1 = map.gradient['ro']
            map.flux(xo=0, yo=barr, ro=r + r * tiny)
            f2 = map.gradient['ro']
            res[ll] = 0.5 * (f1 + f2)
        else:
            raise ValueError("Invalid derivative name.")
    return res


def StarryD(barr, r, lmax, d='b'):
    """Compute dF/d{b,r} for each degree with starry.grad."""
    map = starry.grad.Map(lmax)
    res = np.zeros((lmax + 1, len(barr)))
    for ll in range(lmax + 1):
        map.reset()
        for mm in range(-ll, ll + 1):
            map[ll, mm] = 1
        map.flux(xo=0, yo=barr, ro=r)
        if d == 'b':
            res[ll] = map.gradient['yo']
        elif d == 'r':
            res[ll] = map.gradient['ro']
        else:
            raise ValueError("Invalid derivative name.")
    return res


def PaperFigure(larr=[0, 1, 2, 3, 4, 5],
                logdelta=-6, logeps=-12, res=50, d='b'):
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
        dFdb = StarryD(b, r, lmax, d=d)
        dFdb_mp = StarryDExact(b, r, lmax, d=d)
        n = 0
        for l in larr:
            err_rel = np.abs(dFdb[l] - dFdb_mp[l])
            err_frac = np.abs((dFdb[l] - dFdb_mp[l]) /
                              max(1e-9, np.max(np.abs(dFdb_mp[n]))))
            ax[0, i].plot(err_rel, color=cmap(l / (lmax + 2)), lw=1,
                          zorder=-1)
            ax[1, i].plot(err_frac, color=cmap(l / (lmax + 2)), lw=1,
                          zorder=-1)

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
    fig.savefig("stability_grad.pdf", bbox_inches='tight', dpi=300)
    pl.close()


if __name__ == "__main__":
    PaperFigure(logdelta=-6, logeps=-12, d='b')
