"""Test the gradients in the `System` class."""
from starry.kepler import Primary, Secondary, System
import numpy as np
import matplotlib.pyplot as pl


def test_gradients(plot=False):
    """Test the gradients in the `System` class."""
    # Limb-darkened star
    A = Primary(lmax=3)
    A[1] = 0.4
    A[2] = 0.26
    A[3] = -0.25
    A.r_m = 1e11

    # Dipole-map hot jupiter
    b = Secondary(lmax=2)
    b.r = 0.09
    b.a = 60
    b.inc = 89.943
    b.porb = 50
    b.prot = 2.49
    b.lambda0 = 89.9
    b.ecc = 0.3
    b.w = 89
    b.L = 1.75e-3
    b[1, 0] = 0.5
    b[2, 1] = 0.1
    b[2, 2] = -0.05

    # Dipole-map hot jupiter
    c = Secondary(lmax=1)
    c.r = 0.12
    c.a = 80
    c.inc = 89.95
    c.porb = 100
    c.prot = 7.83
    c.lambda0 = 85
    c.ecc = 0.29
    c.w = 87.4
    c.L = 1.5e-3
    c[1, 0] = 0.4

    # Instantiate the system
    # We're adding a ton of light travel delay
    # and a finite exposure time: this is a
    # comprehensive test of the main `starry` features
    system = System(A, b, c)
    system.exposure_time = 0.02

    # Light curves and gradients of this object
    object = system

    # Let's plot transit, eclipse, and a PPO
    for t1, t2, figname in zip([-0.425, 25.1, -2.6], [0.0, 25.75, -2.0],
                               ["gradients_transit.png",
                                "gradients_eclipse.png",
                                "gradients_ppo.png"]):

        # Time arrays
        time = np.linspace(t1, t2, 500)
        time_num = np.linspace(t1, t2, 50)

        # Set up the plot
        if plot:
            fig = pl.figure(figsize=(6, 10))
            fig.subplots_adjust(hspace=0, bottom=0.05, top=0.95)

        # Run!
        system.compute(time, gradient=True)
        flux = np.array(object.lightcurve)
        grad = dict(object.gradient)

        # Numerical flux
        system.compute(time_num, gradient=True)
        flux_num = np.array(object.lightcurve)

        # Plot it
        if plot:
            ax = pl.subplot2grid((18, 3), (0, 0), rowspan=5, colspan=3)
            ax.plot(time, flux, color='C0')
            ax.plot(time_num, flux_num, 'o', ms=3, color='C1')
            ax.set_yticks([])
            ax.set_xticks([])
            [i.set_linewidth(0.) for i in ax.spines.values()]
            col = 0
            row = 0
        eps = 1e-8
        error_rel = []
        for key in grad.keys():
            if key.endswith('.y') or key.endswith('.u'):
                for i, gradient in enumerate(grad[key]):
                    if plot:
                        axg = pl.subplot2grid((18, 3), (5 + row, col), colspan=1)
                        axg.plot(time, gradient, lw=1, color='C0')
                    if key.endswith('.y'):
                        y0 = eval(key)
                        y = np.array(y0)
                        y[i + 1] += eps
                        exec(key[0] + "[:, :] = y")
                        system.compute(time)
                        exec(key[0] + "[:, :] = y0")
                    else:
                        u0 = eval(key)
                        u = np.array(u0)
                        u[i] += eps
                        exec(key[0] + "[:] = u")
                        system.compute(time)
                        exec(key[0] + "[:] = u0")
                    numgrad = (object.lightcurve - flux) / eps
                    error_rel.append(np.max(abs(numgrad - gradient)))
                    if plot:
                        axg.plot(time, numgrad, lw=1, alpha=0.5, color='C1')
                        axg.set_ylabel(r"$%s_%d$" % (key, i), fontsize=5)
                        axg.margins(None, 0.5)
                        axg.set_xticks([])
                        axg.set_yticks([])
                        [i.set_linewidth(0.) for i in axg.spines.values()]
                        if row < 12:
                            row += 1
                        else:
                            row = 0
                            col += 1
            else:
                if plot:
                    axg = pl.subplot2grid((18, 3), (5 + row, col), colspan=1)
                    axg.plot(time, grad[key], lw=1, color='C0')
                exec(key + " += eps")
                system.compute(time)
                exec(key + " -= eps")
                numgrad = (object.lightcurve - flux) / eps
                error_rel.append(np.max(abs(numgrad - grad[key])))
                if plot:
                    axg.plot(time, numgrad, lw=1, alpha=0.5, color='C1')
                    axg.margins(None, 0.5)
                    axg.set_xticks([])
                    axg.set_yticks([])
                    axg.set_ylabel(r"$%s$" % key, fontsize=5)
                    [i.set_linewidth(0.) for i in axg.spines.values()]
                    if row < 12:
                        row += 1
                    else:
                        row = 0
                        col += 1

        # Generous error tolerance
        assert np.all(np.array(error_rel) < 1e-5)

        # Save the figure
        if plot:
            fig.savefig(figname, bbox_inches='tight', dpi=300)
            pl.close()


if __name__ == "__main__":
    test_gradients(True)
