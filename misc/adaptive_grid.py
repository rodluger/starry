"""Numerical integration by adaptive mesh refinement."""
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.widgets import Slider


class Flux(object):
    """Store the total flux and the visited points."""

    def __init__(self):
        """Initialize."""
        self.total = 0
        self.x = []
        self.y = []


def evaluate(x, y, x0, y0, r0, I):
    """Evaluate the specific intensity I(x, y) at a point."""
    if (x - x0) ** 2 + (y - y0) ** 2 < r0 ** 2:
        return 0
    else:
        return I(x, y)


def fcell(r1, r2, t1, t2, x0, y0, r0, I, flux):
    """Return the flux in a cell."""
    A = 0.5 * (r2 * r2 - r1 * r1) * (t2 - t1)
    r = 0.5 * (r1 + r2)
    t = 0.5 * (t1 + t2)
    f = 0
    for r in [r1, r2]:
        for t in [t1, t2]:
            x = r * np.cos(t)
            y = r * np.sin(t)
            flux.x.append(x)
            flux.y.append(y)
            f += A * evaluate(x, y, x0, y0, r0, I)
    return f / 4


def fnum(r1, r2, t1, t2, x0, y0, r0, I, tol, flux):
    """Return the numerically computed flux."""
    # Coarse estimate
    fcoarse = fcell(r1, r2, t1, t2, x0, y0, r0, I, flux)

    # Fine estimate
    r = 0.5 * (r1 + r2)
    t = 0.5 * (t1 + t2)
    ffine = (fcell(r1, r, t1, t, x0, y0, r0, I, flux) +
             fcell(r1, r, t, t2, x0, y0, r0, I, flux) +
             fcell(r, r2, t1, t, x0, y0, r0, I, flux) +
             fcell(r, r2, t, t2, x0, y0, r0, I, flux))

    # Compare
    if (np.abs(fcoarse - ffine) > tol):
        # Recurse
        fnum(r1, r, t1, t, x0, y0, r0, I, tol, flux)
        fnum(r1, r, t, t2, x0, y0, r0, I, tol, flux)
        fnum(r, r2, t1, t, x0, y0, r0, I, tol, flux)
        fnum(r, r2, t, t2, x0, y0, r0, I, tol, flux)
    else:
        flux.total += ffine
        return


def compute(I=lambda x, y: x ** 3 + y ** 2, tol=1e-4):
    """Compute and plot the result."""
    # Plot
    pl.figure(figsize=(5, 6))
    ax = pl.axes([0.2, 0.25, 0.65, 0.65])
    ax.set_title("")
    scat = ax.scatter([], [], color='k', s=2, alpha=0.3)
    x = np.linspace(-1 + 1e-5, 1 - 1e-5, 1000)
    ax.plot(x, - np.sqrt(1 - x ** 2), 'k-')
    ax.plot(x, + np.sqrt(1 - x ** 2), 'k-')
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_aspect(1)
    ax.axis('off')

    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    def Icirc(x, y):
        res = I(x, y)
        res[x ** 2 + y ** 2 > 1] = np.nan
        return res

    ax.imshow(Icirc(X, Y), extent=(-1, 1, -1, 1), cmap='plasma', alpha=0.5)

    x = np.linspace(-0.5 + 1e-5, 0.5 - 1e-5, 1000)
    ohi, = ax.plot(x, - np.sqrt(0.5 ** 2 - x ** 2), 'r-')
    olo, = ax.plot(x, + np.sqrt(0.5 ** 2 - x ** 2), 'r-')

    axx0 = pl.axes([0.2, 0.15, 0.65, 0.03])
    axy0 = pl.axes([0.2, 0.10, 0.65, 0.03])
    axr0 = pl.axes([0.2, 0.05, 0.65, 0.03])
    sx0 = Slider(axx0, 'x0', -1.0, 1.0, valinit=0.0)
    sy0 = Slider(axy0, 'y0', -1.0, 1.0, valinit=0.0)
    sr0 = Slider(axr0, 'r0', 0.01, 1.0, valinit=0.5)

    def update(val=None):
        flux = Flux()
        b = np.sqrt(sx0.val ** 2 + sy0.val ** 2)
        theta = np.arctan2(sy0.val, sx0.val)
        if b > 1 + sr0.val:
            fnum(0, 1, theta, theta + 2 * np.pi, sx0.val, sy0.val,
                 sr0.val, I, tol / np.pi, flux)
        elif b > 1:
            fnum(0, (1 + b - sr0.val) / 2., theta, theta + 2 * np.pi,
                 sx0.val, sy0.val, sr0.val, I, tol / np.pi, flux)
            fnum((1 + b - sr0.val) / 2., 1, theta, theta + 2 * np.pi,
                 sx0.val, sy0.val, sr0.val, I, tol / np.pi, flux)
        else:
            fnum(0, b, theta, theta + 2 * np.pi, sx0.val, sy0.val,
                 sr0.val, I, tol / np.pi, flux)
            fnum(b, 1, theta, theta + 2 * np.pi, sx0.val, sy0.val,
                 sr0.val, I, tol / np.pi, flux)
        scat.set_offsets(list(zip(flux.x, flux.y)))

        x = np.linspace(sx0.val - sr0.val + 1e-5,
                        sx0.val + sr0.val - 1e-5, 1000)
        ohi.set_xdata(x)
        ohi.set_ydata(sy0.val + np.sqrt(sr0.val ** 2 - (x - sx0.val) ** 2))
        olo.set_xdata(x)
        olo.set_ydata(sy0.val - np.sqrt(sr0.val ** 2 - (x - sx0.val) ** 2))
        ax.set_title("Flux: %.5f" % flux.total)

    sx0.on_changed(update)
    sy0.on_changed(update)
    sr0.on_changed(update)
    update()

    pl.show()


if __name__ == "__main__":
    # Go!
    compute()
