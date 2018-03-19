"""Test various numerical issues."""
import numpy as np
import matplotlib.pyplot as pl
import starry


def smallb():
    """[FIXED] Test numerical error close to b = 0."""
    lmax = 9
    ylm = starry.Map(lmax)

    # Set up the plot
    fig, ax = pl.subplots(lmax, lmax, figsize=(9, 7))
    for axis in ax.flatten():
        for direction in ['top', 'bottom', 'left', 'right']:
            axis.spines[direction].set_linewidth(0.1)
        axis.set_xlim(-15, 0)
        axis.set_ylim(-30, 0)
        axis.set_xticks([-15, -10, -5, 0])
        axis.set_yticks([-30, -20, -10, 0])
        axis.xaxis.set_tick_params(width=0.1)
        axis.yaxis.set_tick_params(width=0.1)
        if axis != ax[-1, 0]:
            axis.set_xticklabels([])
            axis.set_yticklabels([])
        else:
            for tick in axis.get_xticklabels() + axis.get_yticklabels():
                tick.set_fontsize(4)
            axis.set_xlabel(r'$\log\,b$', fontsize=6)
            axis.set_ylabel(r'$\log\,F$', fontsize=6)
        axis.axis('off')

    # Load the benchmarked no Taylor expansion result
    notaylor = np.loadtxt("smallb_notaylor.dat").reshape(lmax, lmax, 1000)

    # Occultation params
    yo = 0
    ro = 0.01
    logxo = np.linspace(-15, 0, 1000)
    for i, l in enumerate(range(1, lmax + 1)):
        for j, m in enumerate(range(1, l + 1)):
            ax[i, j].axis('on')

            # Plot the flux w/out the Taylor expansion
            logflux_notaylor = notaylor[i, j]
            ax[i, j].plot(logxo, logflux_notaylor, color='C0')

            # Plot the flux with the Taylor expansion
            ylm.reset()
            ylm.set_coeff(l, m, 1)
            flux = ylm.flux(xo=10 ** logxo, yo=yo, ro=ro)
            logflux = np.log10(np.abs(flux))
            ax[i, j].plot(logxo, logflux, color='C1')

    # Hack a legend
    axleg = pl.axes([0.7, 0.7, 0.15, 0.15])
    axleg.plot([0, 0], [1, 1], label=r'Recursion', color='C0')
    axleg.plot([0, 0], [1, 1], label=r'Taylor Expansion', color='C1')
    axleg.axis('off')
    leg = axleg.legend(title=r'Flux at small b for m > 0', fontsize=12)
    leg.get_title().set_fontsize('14')
    leg.get_frame().set_linewidth(0.0)

    pl.show()


def mandelagol():
    """[BROKEN] Numerical error as b --> r."""
    ylm = starry.Map(2)
    ylm[1, 0] = 1
    xo = np.linspace(0.5 - 1e-5, 0.5 + 1e-5, 1000)
    flux = ylm.flux(xo=xo, yo=0, ro=0.5)
    pl.plot(xo, flux)
    pl.show()


if __name__ == "__main__":
    smallb()
