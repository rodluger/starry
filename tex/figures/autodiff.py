"""Test autodiff with the `System` class."""
from starry2.kepler import Primary, Secondary, System
import numpy as np
import matplotlib.pyplot as pl


def partial(x):
    """Return the LaTeX form of the partial deriv of x with respect to F."""
    return r'$\frac{\partial F}{\partial %s}$' % x


# Time arrays
time_transit = np.linspace(-0.3, 0.2, 2500)
time_secondary = np.linspace(24.75, 25.5, 2500)

# Limb-darkened star
star = Primary()
star[1] = 0.4
star[2] = 0.26
star.r_m = 0

# Dipole-map hot jupiter
planet = Secondary()
planet.r = 0.1
planet.a = 60
planet.inc = 89.5
planet.porb = 50
planet.prot = 2.49
planet.lambda0 = 89.9
planet.ecc = 0.3
planet.w = 89
planet.L = 1e-3
planet[1, 0] = 0.5

# Instantiate the system
system = System(star, planet)
system.exposure_time = 0

# Set up the plot
fig = pl.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0, bottom=0.05, top=0.95)

# Compute the flux during transit and during secondary eclipse
titles = ['Transit', 'Secondary Eclipse']
for i, time in enumerate([time_transit, time_secondary]):

    # Run!
    system.compute(time, gradient=True)
    flux = np.array(system.lightcurve)
    grad = dict(system.gradient)

    # Plot it
    ax = pl.subplot2grid((20, 2), (0, i), rowspan=5)
    ax.set_title(titles[i])
    ax.plot(time, system.lightcurve, color='C0')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel('Flux', fontsize=16, labelpad=7)
    [i.set_linewidth(0.) for i in ax.spines.values()]

    # Now plot selected gradients
    params = ['time', 'b.r', 'b.L', 'b.porb',
              'b.ecc', 'b.inc', 'b.w',
              'b.prot',
              'b.Y_{1,0}', 'b.Y_{1,1}', 'b.Y_{2,0}',
              'b.Y_{2,1}', 'b.Y_{2,2}',
              'A.u_{1}',  'A.u_{2}']

    labels = ['t', 'r', 'L', 'P',
              'e', 'i', r'\omega', 'P_r',
              'Y_{1,0}', 'Y_{1,1}', 'Y_{2,0}',
              'Y_{2,1}', 'Y_{2,2}', 'u_{1}', 'u_{2}']

    for param, label, n in zip(params, labels, range(5, 5 + len(labels))):
        axg = pl.subplot2grid((20, 2), (n, i))
        axg.plot(time, grad[param], lw=1, color='C1')
        axg.margins(None, 0.5)
        axg.set_xticks([])
        axg.set_yticks([])
        axg.set_ylabel(partial(label), rotation=0, fontsize=14)
        axg.yaxis.set_label_coords(-0.07, 0.05)
        [i.set_linewidth(0.) for i in axg.spines.values()]

        # Compute and plot the numerical gradient
        eps = 1e-8
        if param == 'time':
            system.compute(time + eps)
        elif param == 'b.r':
            planet.r += eps
            system.compute(time)
            planet.r -= eps
        elif param == 'b.L':
            planet.L += eps
            system.compute(time)
            planet.L -= eps
        elif param == 'b.porb':
            planet.porb += eps
            system.compute(time)
            planet.porb -= eps
        elif param == 'b.ecc':
            planet.ecc += eps
            system.compute(time)
            planet.ecc -= eps
        elif param == 'b.inc':
            planet.inc += eps
            system.compute(time)
            planet.inc -= eps
        elif param == 'b.w':
            planet.w += eps
            system.compute(time)
            planet.w -= eps
        elif param == 'b.prot':
            planet.prot += eps
            system.compute(time)
            planet.prot -= eps
        elif 'Y_' in param:
            l, m = int(param[5]), int(param[7])
            planet[l, m] += eps
            system.compute(time)
            planet[l, m] -= eps
        elif param == 'A.u_{1}':
            star[1] += eps
            system.compute(time)
            star[1] -= eps
        elif param == 'A.u_{2}':
            star[2] += eps
            system.compute(time)
            star[2] -= eps
        numgrad = (system.lightcurve - flux) / eps
        axg.plot(time[::100], numgrad[::100], lw=0, ms=2,
                 marker='.', color='C1')

    axg.set_xlabel('Time', fontsize=16)

fig.savefig('autodiff.pdf', bbox_inches='tight')
