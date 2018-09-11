"""Stellar transit example."""
from starry2 import Map
import matplotlib.pyplot as pl
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import curve_fit


def IofMu(mu, *u):
    """Specific intensity."""
    return (1 - np.sum([u[l] * (1 - mu) ** l
                        for l in range(1, len(u))], axis=0))


def NumericalFlux(b, r, u):
    """Compute the flux by numerical integration of the surface integral."""
    # I'm only coding up a specific case here
    assert (b >= 0) and (r <= 1), "Invalid range."

    if b >= 1 + r:
        return 1

    # Get points of intersection
    if b > 1 - r:
        yi = (1. + b ** 2 - r ** 2) / (2. * b)
        xi = (1. / (2. * b)) * np.sqrt(4 * b ** 2 - (1 + b ** 2 - r ** 2) ** 2)
    else:
        yi = np.inf
        xi = r

    # Specific intensity map
    norm = np.pi * (1 - 2 * np.sum([u[l] / ((l + 1) * (l + 2))
                                    for l in range(1, len(u))]))

    def I(y, x):
        mu = np.sqrt(1 - x ** 2 - y ** 2)
        return (1 - np.sum([u[l] * (1 - mu) ** l
                            for l in range(1, len(u))])) / norm

    # Lower integration limit
    def y1(x):
        if yi <= b:
            # Lower occultor boundary
            return b - np.sqrt(r ** 2 - x ** 2)
        elif b <= 1 - r:
            # Lower occultor boundary
            return b - np.sqrt(r ** 2 - x ** 2)
        else:
            # Tricky: we need to do this in two parts
            return b - np.sqrt(r ** 2 - x ** 2)

    # Upper integration limit
    def y2(x):
        if yi <= b:
            # Upper occulted boundary
            return np.sqrt(1 - x ** 2)
        elif b <= 1 - r:
            # Upper occultor boundary
            return b + np.sqrt(r ** 2 - x ** 2)
        else:
            # Tricky: we need to do this in two parts
            return np.sqrt(1 - x ** 2)

    # Compute the total flux
    flux, _ = dblquad(I, -xi, xi, y1, y2, epsabs=1e-14, epsrel=1e-14)

    # Do we need to solve an additional integral?
    if not (yi <= b) and not (b <= 1 - r):

        def y1(x):
            return b - np.sqrt(r ** 2 - x ** 2)

        def y2(x):
            return b + np.sqrt(r ** 2 - x ** 2)

        additional_flux, _ = dblquad(I, -r, -xi, y1, y2,
                                     epsabs=1e-14, epsrel=1e-14)

        flux += 2 * additional_flux

    return 1 - flux


# Setup
fig, ax = pl.subplots(2, figsize=(5, 6))
fig.subplots_adjust(left=0.15, hspace=0.3)

# Let's define a (synthetic) grid of specific intensities
mu = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
I = 1 + np.log10(0.9 * mu + 0.1)
ax[1].plot(mu, I, 'ko')

# Let's fit for the quadratic limb-darkening coefficients
u_quad, _ = curve_fit(IofMu, mu, I, np.zeros(3))

# Now let's fit for the coefficients that fit our model *exactly*
u_exact, _ = curve_fit(IofMu, mu, I, np.zeros_like(mu))

# Plot our two models
mu_hires = np.linspace(0, 1, 100)
ax[1].plot(mu_hires, IofMu(mu_hires, *u_quad), label='Quadratic')
ax[1].plot(mu_hires, IofMu(mu_hires, *u_exact), label='Exact')

# Compute and plot the starry flux in transit for both models
npts = 500
r = 0.1
b = np.linspace(-1.5, 1.5, npts)
for u, label in zip([u_quad, u_exact], ['Quadratic', 'Exact']):
    map = Map(len(u) - 1)
    for l in range(1, len(u)):
        map[l] = u[l]
    sF = map.flux(xo=b, yo=0, ro=r)
    ax[0].plot(b, sF, '-')

# Appearance
ax[0].set_xlim(-1.5, 1.5)
ax[0].set_ylabel('Normalized flux', fontsize=16)
ax[0].set_xlabel('Impact parameter', fontsize=16)
ax[1].set_ylabel('Specific Intensity', fontsize=16, labelpad=10)
ax[1].legend(loc='lower left')
ax[1].invert_xaxis()
ax[1].set_xlim(1, 0)
ax[1].set_xlabel(r'$\mu$', fontsize=16)

# Save
pl.savefig('high_order_ld.pdf', bbox_inches='tight')
