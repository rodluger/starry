"""Test PPO autodiff with the `System` class."""
from starry.kepler import Primary, Secondary, System
import numpy as np
import matplotlib.pyplot as pl


time = np.linspace(-2.6, -2.0, 500)

# Limb-darkened A
A = Primary(lmax=2)
A[1] = 0.4
A[2] = 0.26
A.r_m = 0

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
system = System(A, b, c)
system.exposure_time = 0

# Light curves and gradients of this object
object = system

# Set up the plot
fig = pl.figure(figsize=(6, 10))
fig.subplots_adjust(hspace=0, bottom=0.05, top=0.95)

# Run!
system.compute(time, gradient=True)
flux = np.array(object.lightcurve)
grad = dict(object.gradient)

# Plot it
ax = pl.subplot2grid((18, 3), (0, 0), rowspan=5, colspan=3)
ax.plot(time, object.lightcurve, color='C0')
ax.set_yticks([])
ax.set_xticks([])
[i.set_linewidth(0.) for i in ax.spines.values()]
col = 0
row = 0
eps = 1e-8
for key in grad.keys():
    if key.endswith('.y') or key.endswith('.u'):
        for i, gradient in enumerate(grad[key]):
            axg = pl.subplot2grid((18, 3), (5 + row, col), colspan=1)
            axg.plot(time, gradient, lw=1, color='C1')
            if key.endswith('.y'):
                y0 = eval(key)
                y = np.array(y0)
                y[i + 1] += eps
                exec(key[0] + "[:, :] = y")
                system.compute(time)
                exec(key[0] + "[:, :] = y0")
                numgrad = (object.lightcurve - flux) / eps
                axg.plot(time, numgrad, lw=1, alpha=0.5, color='C0')
                axg.set_ylabel(r"$%s_%d$" % (key, i + 1), fontsize=5)
            else:
                u0 = eval(key)
                u = np.array(u0)
                u[i] += eps
                exec(key[0] + "[:] = u")
                system.compute(time)
                exec(key[0] + "[:] = u0")
                numgrad = (object.lightcurve - flux) / eps
                axg.plot(time, numgrad, lw=1, alpha=0.5, color='C0')
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
        axg = pl.subplot2grid((18, 3), (5 + row, col), colspan=1)
        axg.plot(time, grad[key], lw=1, color='C1')
        exec(key + " += eps")
        system.compute(time)
        exec(key + " -= eps")
        numgrad = (object.lightcurve - flux) / eps
        axg.plot(time, numgrad, lw=1, alpha=0.5, color='C0')
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

fig.savefig('autodiff.pdf', bbox_inches='tight')
