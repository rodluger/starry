"""Geometry of occultation."""
import numpy as np
import matplotlib.pyplot as pl

# Set up
fig, ax = pl.subplots(1, figsize=(5, 6))
fig.subplots_adjust(left=0, right=1)
ax.set_aspect(1)
ax.axis('off')

# Occulted body
x = np.linspace(-1, 1, 1000)
y1 = -np.sqrt(1 - x ** 2)
y2 = np.sqrt(1 - x ** 2)
ax.plot(x, y1, 'k-', lw=0.5)
ax.plot(x, y2, 'k-', lw=0.5)
ax.fill_between(x, -np.sqrt(1 - x ** 2),
                np.sqrt(1 - x ** 2), color='#f5f5f5')

# Occultor
r = 0.75
b = 1.25
sinphi = (1 - r ** 2 - b ** 2) / (2 * b * r)
cosphi = np.cos(np.arcsin(sinphi))
x = np.linspace(-r, r, 1000)
y1 = b - np.sqrt(r ** 2 - x ** 2)
y2 = b + np.sqrt(r ** 2 - x ** 2)
ax.plot(x, y1, 'r-', lw=0.5)
ax.plot(x, y2, 'r-', lw=0.5)

# Fill the visible area
x = np.linspace(-r, r, 1000)
ax.fill_between(x, b - np.sqrt(r ** 2 - x ** 2),
                b + np.sqrt(r ** 2 - x ** 2), color='#fff5f5')

# Draw the lines
ax.plot([0, 1], [0, 0], 'k:', lw=0.5)
ax.plot([0, r], [b, b], 'r:', lw=0.5)
ax.plot([0, sinphi], [b, cosphi], 'r-', lw=0.5)
ax.plot([0, -sinphi], [b, cosphi], 'r-', lw=0.5)
ax.plot([0, 0], [0, b], 'k:', lw=0.5)
x = np.linspace(sinphi, -sinphi, 1000)
ax.plot(x, b - np.sqrt(r ** 2 - x ** 2), 'r-', lw=2)
ax.plot(x, -np.sqrt(1 - x ** 2), 'k-', lw=2)
x = np.linspace(-1, sinphi, 1000)
ax.plot(x, -np.sqrt(1 - x ** 2), 'k-', lw=2)
ax.plot(x, np.sqrt(1 - x ** 2), 'k-', lw=2)
x = np.linspace(-sinphi, 1, 1000)
ax.plot(x, -np.sqrt(1 - x ** 2), 'k-', lw=2)
ax.plot(x, np.sqrt(1 - x ** 2), 'k-', lw=2)
ax.plot([0, -sinphi], [0, cosphi], 'k-', lw=0.5)
ax.plot([0, sinphi], [0, cosphi], 'k-', lw=0.5)

# Label the lines
ax.annotate(r"$1$", xy=(0.5, -0.12), xycoords="data", xytext=(0, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=12, color="k")
ax.annotate(r"$r$", xy=(r / 2, b + 0.1), xycoords="data", xytext=(0, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=12, color="r")
ax.annotate(r"$b$", xy=(-0.13, 0.6), xycoords="data", xytext=(0, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=12, color="k")

# Label the angles
x = np.linspace(0.15, 0.1875, 1000)
y = b - np.sqrt(0.1875 ** 2 - x ** 2)
ax.plot(x, y, 'r-', lw=0.5, zorder=-1)
ax.annotate(r"-$\phi$", xy=(0.2, 1.16), xycoords="data", xytext=(0, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=12, color="r")
ax.annotate(r"$\pi-\phi$", xy=(sinphi - 0.15, cosphi + 0.075), xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=12, color="r")
ax.annotate(r"$\pi-\lambda$", xy=(sinphi - 0.175, cosphi - 0.075),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=12, color="k")
ax.annotate(r"$2\pi+\phi$", xy=(-sinphi + 0.175, cosphi + 0.075),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=12, color="r")
ax.annotate(r"$2\pi+\lambda$", xy=(-sinphi + 0.175, cosphi - 0.075),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=12, color="k")
x = np.linspace(0.115, 0.1875, 1000)
y = np.sqrt(0.1875 ** 2 - x ** 2)
ax.plot(x, y, 'k-', lw=0.5, zorder=1)
ax.annotate(r"$\lambda$", xy=(0.2, 0.1), xycoords="data", xytext=(0, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=12, color="k")

# Draw the dots
ax.plot(0, 0, 'ko', ms=3)
ax.plot(0, b, 'ro', ms=3)
ax.plot(-sinphi, cosphi, 'ro', ms=3)
ax.plot(sinphi, cosphi, 'ro', ms=3)

# Label the bodies
ax.annotate(r"$\mathrm{occultor}$", xy=(0, b + r + 0.25), xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=10, color="r")
ax.annotate(r"$\mathrm{(rotated\ frame)}$", xy=(0, b + r + 0.1),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=10, color="r")
ax.plot([0], [b+r+0.3], '.', ms=0)
ax.annotate(r"$\mathrm{occulted}$", xy=(0, -1.1), xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color="k")
ax.annotate(r"$\mathrm{(rotated\ frame)}$", xy=(0, -1.25),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color="k")

# Occultor (original location)
# Theta is about 124 degrees
r = 0.75
y0 = -0.7
x0 = np.sqrt(b ** 2 - y0 ** 2)
x = np.linspace(x0 - r, x0 + r, 1000)
y1 = y0 - np.sqrt(r ** 2 - (x - x0) ** 2)
y2 = y0 + np.sqrt(r ** 2 - (x - x0) ** 2)
ax.plot(x, y1, 'r-', lw=0.5, alpha=0.5)
ax.plot(x, y2, 'r-', lw=0.5, alpha=0.5)
ax.plot(x0, y0, 'ro', ms=3, alpha=0.5)
ax.plot([x0, 0], [y0, 0], 'k:', lw=0.5, alpha=0.25)
ax.annotate(r"$(x_0, y_0)$", xy=(x0, y0), xycoords="data",
            xytext=(0, -4),
            textcoords="offset points", ha="center", va="top",
            fontsize=12, color="r", alpha=0.5)
x = np.linspace(0, 0.075, 1000)
y = np.sqrt(0.075 ** 2 - x ** 2)
ax.plot(x, y, 'r-', lw=0.5, zorder=1)
x = np.linspace(0.075, 0.063, 1000)
y = -np.sqrt(0.075 ** 2 - x ** 2)
ax.plot(x, y, 'r-', lw=0.5, zorder=1)
ax.annotate(r"$\theta$", xy=(0, -0.05), xycoords="data", xytext=(0, 0),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color="r")
ax.plot([x0], [y0 - r - 0.3], '.', ms=0)
ax.annotate(r"$\mathrm{occultor}$", xy=(x0, y0 - r - 0.1), xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color="r", alpha=0.5)
ax.annotate(r"$\mathrm{(original\ frame)}$", xy=(x0, y0 - r - 0.25),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color="r", alpha=0.5)

fig.savefig('geometry.pdf')
