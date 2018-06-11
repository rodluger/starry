"""Plot some Ylm thermal phase curves."""
import matplotlib.pyplot as pl
import numpy as np
from starry import Map

# Compute and plot up to this order
lmax = 6

# Number of points in the phase curve
nt = 100

# Number of points in the numerical light curve
nn = 10

# Set up the plot
fig, ax = pl.subplots(lmax + 1, lmax + 1, figsize=(9, 5.5))
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
for j, m in enumerate(range(lmax + 1)):
    ax[-1, j].set_xlabel(r"$m = %d$" % m, labelpad=30, fontsize=12)

# Rotate about this vector
ux = np.array([1., 0., 0.])
uy = np.array([0., 1., 0.])
y = Map(lmax)
theta = np.linspace(0, 360, nt, endpoint=False)
thetan = np.linspace(0, 360, nn, endpoint=False)
for i, l in enumerate(range(lmax + 1)):
    for j, m in enumerate(range(l + 1)):
        nnull = 0
        for axis, zorder, color in zip([ux, uy], [1, 0], ['C0', 'C1']):
            y.reset()
            y.set_coeff(l, m, 1)
            flux = y.flux(axis=axis, theta=theta)
            ax[i, j].plot(theta, flux, lw=1, zorder=zorder, color=color)
            fluxn = y._flux_numerical(axis=axis, theta=thetan, tol=1e-5)
            ax[i, j].plot(thetan, fluxn, '.', ms=2, zorder=zorder, color=color)
            if np.max(np.abs(flux)) < 1e-10:
                nnull += 1
        # If there's no light curve, make sure our plot range
        # isn't too tight, as it will zoom in on floating point error
        if nnull == 2:
            ax[i, j].set_ylim(-1, 1)
# Force the scale for the constant map
ax[0, 0].set_ylim(0.886 + 1, 0.886 - 1)

# Hack a legend
axleg = pl.axes([0.7, 0.7, 0.15, 0.15])
axleg.plot([0, 0], [1, 1], label=r'$\vec{u} = \hat{x}$')
axleg.plot([0, 0], [1, 1], label=r'$\vec{u} = \hat{y}$')
axleg.axis('off')
leg = axleg.legend(title=r'Phase curves', fontsize=18)
leg.get_title().set_fontsize('20')
leg.get_frame().set_linewidth(0.0)

# Save!
fig.savefig("ylmphasecurves.pdf", bbox_inches='tight')
pl.close(fig)
