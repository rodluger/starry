"""Numerical integration by adaptive mesh refinement."""
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.widgets import Slider
from matplotlib.patches import Arc, ConnectionPatch


class Flux(object):
    """Store the total flux and the visited points."""

    def __init__(self):
        """Initialize."""
        self.total = 0
        self.x = []
        self.y = []
        self.r1 = []
        self.r2 = []
        self.t1 = []
        self.t2 = []


def evaluate(x, y, x0, y0, r0, I):
    """Evaluate the specific intensity I(x, y) at a point."""
    if (x - x0) ** 2 + (y - y0) ** 2 < r0 ** 2:
        return 0
    else:
        return I(x, y)


def fcell(r1, r2, t1, t2, x0, y0, r0, I, flux, logpts=False):
    """Return the flux in a cell."""
    deltheta = np.abs((t1 + np.pi - t2) % (2 * np.pi) - np.pi)
    A = 0.5 * (r2 * r2 - r1 * r1) * deltheta
    f = 0
    for r in [r1, r2]:
        for t in [t1, t2]:
            x = r * np.cos(t)
            y = r * np.sin(t)
            f += A * evaluate(x, y, x0, y0, r0, I)
    if logpts:
        # Log the midpoint for plotting
        r = 0.5 * (r1 + r2)
        num = (np.sin(t1) + np.sin(t2))
        den = (np.cos(t1) + np.cos(t2))
        nds = np.sqrt(den ** 2 + num ** 2)
        x = r * den / nds
        y = r * num / nds
        flux.r1.append(r1)
        flux.r2.append(r2)
        flux.t1.append(t1)
        flux.t2.append(t2)
        flux.x.append(x)
        flux.y.append(y)
    return f / 4


def fnum(r1, r2, t1, t2, x0, y0, r0, I, tol, flux):
    """Return the numerically computed flux."""
    # Coarse estimate
    fcoarse = fcell(r1, r2, t1, t2, x0, y0, r0, I, flux, True)

    # Fine estimate
    r = 0.5 * (r1 + r2)
    t = np.arctan2(np.sin(t1) + np.sin(t2), np.cos(t1) + np.cos(t2))

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


def compute(I=lambda x, y: x ** 2 + y ** 2, tol=1e-4, show_cells=False):
    """Compute and plot the result."""
    # Plot
    fig = pl.figure(figsize=(5, 6))
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

    # Initial occultor position
    x0 = 0.
    y0 = -1
    r0 = 0.1
    x = np.linspace(x0 - r0 + 1e-5, x0 + r0 - 1e-5, 1000)
    ohi, = ax.plot(x, y0 - np.sqrt(r0 ** 2 - (x - x0) ** 2), 'r-')
    olo, = ax.plot(x, y0 + np.sqrt(r0 ** 2 - (x - x0) ** 2), 'r-')

    axx0 = pl.axes([0.2, 0.15, 0.65, 0.03])
    axy0 = pl.axes([0.2, 0.10, 0.65, 0.03])
    axr0 = pl.axes([0.2, 0.05, 0.65, 0.03])
    sx0 = Slider(axx0, 'x0', -1.0, 1.0, valinit=x0)
    sy0 = Slider(axy0, 'y0', -1.0, 1.0, valinit=y0)
    sr0 = Slider(axr0, 'r0', 0.01, 1.0, valinit=r0)

    def update(val=None):
        flux = Flux()
        b = np.sqrt(sx0.val ** 2 + sy0.val ** 2)
        theta = np.arctan2(sy0.val, sx0.val)
        theta0 = theta - np.pi
        if theta0 < 0:
            theta0 += 2 * np.pi
        elif theta0 > 2 * np.pi:
            theta0 -= 2 * np.pi

        if b <= sr0.val:
            rmid = 0.5
            deltheta = 0.5
        elif b > 1 + sr0.val:
            rmid = 0.5
            deltheta = 0.5
        elif b > 1:
            rmid = (1 + b - sr0.val) / 2.
            deltheta = 0.95 * np.abs(np.arccos((b ** 2 -
                                                sr0.val ** 2 + rmid ** 2) /
                                               (2 * b * rmid)))
        else:
            rmid = b
            deltheta = 0.95 * np.abs(np.arccos(1 - 0.5 * (sr0.val / b) ** 2))

        theta1 = theta - deltheta
        if theta1 < 0:
            theta1 += 2 * np.pi
        theta2 = theta + deltheta
        if theta1 > 2 * np.pi:
            theta2 -= 2 * np.pi

        # A
        fnum(0, rmid, theta1, theta2,
             sx0.val, sy0.val, sr0.val, I, tol / np.pi, flux)
        # B
        fnum(rmid, 1, theta1, theta2,
             sx0.val, sy0.val, sr0.val, I, tol / np.pi, flux)
        # C
        fnum(0, rmid, theta0, theta1,
             sx0.val, sy0.val, sr0.val, I, tol / np.pi, flux)
        # D
        fnum(rmid, 1, theta0, theta1,
             sx0.val, sy0.val, sr0.val, I, tol / np.pi, flux)
        # E
        fnum(0, rmid, theta0, theta2,
             sx0.val, sy0.val, sr0.val, I, tol / np.pi, flux)
        # F
        fnum(rmid, 1, theta0, theta2,
             sx0.val, sy0.val, sr0.val, I, tol / np.pi, flux)

        scat.set_offsets(list(zip(flux.x, flux.y)))
        x = np.linspace(sx0.val - sr0.val + 1e-5,
                        sx0.val + sr0.val - 1e-5, 1000)
        ohi.set_xdata(x)
        ohi.set_ydata(sy0.val + np.sqrt(sr0.val ** 2 - (x - sx0.val) ** 2))
        olo.set_xdata(x)
        olo.set_ydata(sy0.val - np.sqrt(sr0.val ** 2 - (x - sx0.val) ** 2))
        ax.set_title("Flux: %.5f" % flux.total)

        # Plot the cell boundaries
        if show_cells:
            for patch in ax.patches:
                patch.remove()
            ax.patches = []
            fig.canvas.draw_idle()
            for r1, r2, t1, t2 in zip(flux.r1, flux.r2, flux.t1, flux.t2):
                t1d = t1 * 180 / np.pi
                t2d = t2 * 180 / np.pi
                arc1 = Arc((0, 0), 2 * r1, 2 * r1, theta1=t1d,
                           theta2=t2d, lw=0.5)
                arc2 = Arc((0, 0), 2 * r2, 2 * r2, theta1=t1d,
                           theta2=t2d, lw=0.5)
                x1 = r1 * np.cos(t1)
                y1 = r1 * np.sin(t1)
                x2 = r2 * np.cos(t1)
                y2 = r2 * np.sin(t1)
                rad1 = ConnectionPatch((x1, y1), (x2, y2), "data",
                                       "data", lw=0.5)
                x1 = r1 * np.cos(t2)
                y1 = r1 * np.sin(t2)
                x2 = r2 * np.cos(t2)
                y2 = r2 * np.sin(t2)
                rad2 = ConnectionPatch((x1, y1), (x2, y2), "data",
                                       "data", lw=0.5)
                ax.add_patch(arc1)
                ax.add_patch(arc2)
                ax.add_patch(rad1)
                ax.add_patch(rad2)

    sx0.on_changed(update)
    sy0.on_changed(update)
    sr0.on_changed(update)
    update()

    pl.show()


if __name__ == "__main__":
    # Go!
    compute(show_cells=False, tol=1e-4)
