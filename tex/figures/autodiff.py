"""Test autodiff with the `System` class."""
from starry.grad import Star, Planet, System
import numpy as np
import matplotlib.pyplot as pl


def partial(x):
    """Return the LaTeX form of the partial deriv of x with respect to F."""
    return r'$\frac{\partial F}{\partial %s}$' % x


# Time arrays
time_transit = np.linspace(-0.3, 0.2, 2500)
time_secondary = np.linspace(24.75, 25.5, 2500)

# Limb-darkened star
star = Star()
star.map[1] = 0.4
star.map[2] = 0.26

# Dipole-map hot jupiter
planet = Planet(lmax=2, r=0.1, a=60, inc=89.5, porb=50, prot=2.49,
                lambda0=89.9, ecc=0.3, w=89, L=1e-3)
planet.map[1, 0] = 0.5

# Instantiate the system
system = System([star, planet], exposure_time=0)

# Set up the plot
fig = pl.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0, bottom=0.05, top=0.95)

# Compute the flux during transit and during secondary eclipse
titles = ['Transit', 'Secondary Eclipse']
for i, time in enumerate([time_transit, time_secondary]):

    # Run!
    system.compute(time)
    flux = np.array(system.flux)
    grad = dict(system.gradient)

    # Plot it
    ax = pl.subplot2grid((20, 2), (0, i), rowspan=5)
    ax.set_title(titles[i])
    ax.plot(time, system.flux, color='C0')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel('Flux', fontsize=16, labelpad=7)
    [i.set_linewidth(0.) for i in ax.spines.values()]

    # Now plot selected gradients
    params = ['time', 'planet1.r', 'planet1.L', 'planet1.porb',
              'planet1.ecc', 'planet1.inc', 'planet1.w',
              'planet1.prot',
              'planet1.Y_{1,0}', 'planet1.Y_{1,1}', 'planet1.Y_{2,0}',
              'planet1.Y_{2,1}', 'planet1.Y_{2,2}',
              'star.u_1',  'star.u_2']

    labels = ['t', 'r', 'L', 'P',
              'e', 'i', r'\omega', 'P_r',
              'Y_{1,0}', 'Y_{1,1}', 'Y_{2,0}',
              'Y_{2,1}', 'Y_{2,2}', 'u_1', 'u_2']

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
        elif param == 'planet1.r':
            planet.r += eps
            system.compute(time)
            planet.r -= eps
        elif param == 'planet1.L':
            planet.L += eps
            system.compute(time)
            planet.L -= eps
        elif param == 'planet1.porb':
            planet.porb += eps
            system.compute(time)
            planet.porb -= eps
        elif param == 'planet1.ecc':
            planet.ecc += eps
            system.compute(time)
            planet.ecc -= eps
        elif param == 'planet1.inc':
            planet.inc += eps
            system.compute(time)
            planet.inc -= eps
        elif param == 'planet1.w':
            planet.w += eps
            system.compute(time)
            planet.w -= eps
        elif param == 'planet1.prot':
            planet.prot += eps
            system.compute(time)
            planet.prot -= eps
        elif 'Y_' in param:
            l, m = int(param[11]), int(param[13])
            planet.map[l, m] += eps
            system.compute(time)
            planet.map[l, m] -= eps
        elif param == 'star.u_1':
            star.map[1] += eps
            system.compute(time)
            star.map[1] -= eps
        elif param == 'star.u_2':
            star.map[2] += eps
            system.compute(time)
            star.map[2] -= eps
        numgrad = (system.flux - flux) / eps
        axg.plot(time[::100], numgrad[::100], lw=0, ms=2,
                 marker='.', color='C1')

    axg.set_xlabel('Time', fontsize=16)

fig.savefig('autodiff.pdf', bbox_inches='tight')
