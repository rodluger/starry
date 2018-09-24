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


def animate(I, res=300, cmap="plasma", gif="", interval=75, labels=None):
    """Animate the map as it rotates."""
    fig, ax = pl.subplots(1, figsize=(3, 3))
    img = ax.imshow(I[0], origin="lower", interpolation="none", cmap=cmap,
                    extent=(-1, 1, -1, 1), animated=True,
                    vmin=np.nanmin(I), vmax=np.nanmax(I))
    ax.axis('off')
    if labels is not None:
        ax.set_title(labels[0])

    def updatefig(i):
        img.set_array(I[i])
        if labels is not None:
            ax.set_title(labels[i])
            return img, ax
        else:
            return img,

    ani = animation.FuncAnimation(fig, updatefig, interval=interval,
                                  blit=(labels is None),
                                  frames=len(I))

    # Hack to return a gif embedded in HTML if we're
    # inside a Jupyter notebook
    try:
        if 'zmqshell' in str(type(get_ipython())):
            # We're inside a notebook!
            from IPython.display import HTML
            if gif == "":
                gif = str(id(ani))
            elif gif.endswith(".gif"):
                gif = gif[:-4]
            ani.save('%s.gif' % gif, writer='imagemagick')
            pl.close()
            return HTML('<img src="%s.gif">' % gif)
        else:
            raise NameError("")
    except NameError:
        pass

    # Business as usual
    if gif != "":
        if gif.endswith(".gif"):
            gif = gif[:-4]
        ani.save('%s.gif' % gif, writer='imagemagick')
    else:
        pl.show()
    pl.close()
