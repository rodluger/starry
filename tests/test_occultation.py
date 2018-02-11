"""Test the STARRY integration routines."""
import sys; sys.path.insert(1, "../")
from starry import R, A, S
from starry.integrals import brute
import numpy as np
import matplotlib.pyplot as pl


# Occultation parameters
npts = 100
nbrt = 50
lmax = 3
r = 0.3
res = 30
y0 = 0.5
x0hires = np.linspace(-2, 2, npts)
x0lores = np.linspace(-2, 2, nbrt)

# The flux arrays
N = (lmax + 1) ** 2
F = np.zeros((N, npts), dtype=float)
Fbrt = np.zeros((N, nbrt), dtype=float)

# Loop over each Ylm
n = 0
for l in range(lmax + 1):
    for m in range(-l, l + 1):

        print(l, m)

        # Construct the vector for this Ylm
        y = np.zeros(N, dtype=float)
        y[n] = 1

        # Loop through the timeseries
        for i, x0 in enumerate(x0hires):

            # Rotate the map about z to align the occultor with the y axis
            b = np.sqrt(x0 ** 2 + y0 ** 2)
            sintheta = x0 / b
            costheta = y0 / b
            u = [0., 0., 1.]
            RRy = np.dot(R(lmax, u, costheta, sintheta), y)

            # Convert to the Greens basis
            ARRy = np.dot(A(lmax), RRy)

            # Compute the integrals and solve for the flux array
            sT = S(lmax, b, r)
            F[n, i] = np.dot(sT, ARRy)

        # Compute the brute force flux
        for i, x0 in enumerate(x0lores):
            Fbrt[n, i] = brute(y, x0, y0, r, res=res)

        n += 1

# Set up the plot
fig, ax = pl.subplots(lmax + 1, 2 * lmax + 1, figsize=(9, 6))
fig.subplots_adjust(hspace=0)
for axis in ax.flatten():
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
for l in range(lmax + 1):
    ax[l, 0].set_ylabel(r"$l = %d$" % l,
                        rotation='horizontal',
                        labelpad=30, y=0.38,
                        fontsize=12)
for j, m in enumerate(range(-lmax, lmax + 1)):
    if m < 0:
        ax[-1, j].set_xlabel(r"$m {=} \mathrm{-}%d$" % -m,
                             labelpad=30, fontsize=11)
    else:
        ax[-1, j].set_xlabel(r"$m = %d$" % m, labelpad=30, fontsize=11)

# Plot the fluxes
n = 0
for i, l in enumerate(range(lmax + 1)):
    for j, m in enumerate(range(-l, l + 1)):

        # Offset the index for centered plotting
        j += lmax - l

        # Plot the fluxes
        ax[i, j].plot(x0hires, F[n], '-', lw=1)
        ax[i, j].plot(x0lores, Fbrt[n], '.', ms=2)

        # Fix the y limits
        baseline = F[n, -1]
        lim = np.nanmax(np.abs(F[n] - baseline))
        lim = max(1e-10, lim)
        ax[i, j].set_ylim(baseline - 1.1 * lim, baseline + 1.1 * lim)

        n += 1

pl.show()
