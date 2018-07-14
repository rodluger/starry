"""Plotting utilities for starry."""
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import numpy as np


__all__ = ["show", "animate"]


def show(I, res=300, cmap="plasma"):
    """Show the specific intensity I(x, y) on a grid."""
    fig, ax = pl.subplots(1, figsize=(3, 3))
    ax.imshow(I, origin="lower", interpolation="none", cmap=cmap,
              extent=(-1, 1, -1, 1))
    ax.axis('off')
    pl.show()


def animate(I, u=[0, 1, 0], res=300, cmap="plasma"):
    """Animate the map as it rotates about the axis `u`."""
    fig, ax = pl.subplots(1, figsize=(3, 3))
    img = ax.imshow(I[0], origin="lower", interpolation="none", cmap=cmap,
                    extent=(-1, 1, -1, 1), animated=True,
                    vmin=np.nanmin(I), vmax=np.nanmax(I))
    ax.axis('off')

    def updatefig(i):
        img.set_array(I[i])
        return img,

    ani = animation.FuncAnimation(fig, updatefig, interval=75, blit=True,
                                  frames=len(I))
    pl.show()
