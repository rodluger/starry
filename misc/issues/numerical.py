"""Test certain numerical issues."""
import numpy as np
import matplotlib.pyplot as pl
import starry


def smallb1():
    """Numerical error close to b = 0."""
    ylm = starry.Map(9)

    fig, ax = pl.subplots(8, 2, figsize=(6, 8))

    for i, n in enumerate(range(2, 10)):
        ylm.reset()
        ylm[n, n - 1] = 1

        # Linear space
        xo = np.linspace(-1, 1, 10000)
        flux = ylm.flux(xo=xo, yo=0, ro=0.01)
        ax[i, 0].plot(xo, flux, color='C1')

        # Log space
        logxo = np.linspace(-15, 0, 1000)
        flux = ylm.flux(xo=10 ** logxo, yo=0, ro=0.01)
        logflux = np.log10(np.abs(flux))
        ax[i, 1].plot(logxo, logflux, color='C1')

    pl.show()


def mandelagol1():
    """Numerical error when b = 0, r = 1."""
    ylm = starry.Map(2)
    ylm[1, 0] = 1
    logxo = np.linspace(-15, 0, 1000)
    flux = ylm.flux(xo=10 ** logxo, yo=0, ro=1)
    logflux = np.log10(np.abs(flux))
    pl.plot(logxo, logflux)
    pl.show()


if __name__ == "__main__":
    smallb1()
