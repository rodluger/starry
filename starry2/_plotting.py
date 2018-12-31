"""Plotting utilities for starry."""
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import numpy as np
import os


__all__ = ["show", "animate"]


def show(I, res=300, cmap="plasma", gif="", interval=75):
    """Show the specific intensity I(x, y) on a grid."""
    fig, ax = pl.subplots(1, figsize=(3, 3))
    ax.imshow(I.reshape(res, res), origin="lower", 
              interpolation="none", cmap=cmap,
              extent=(-1, 1, -1, 1))
    ax.axis('off')
    pl.show()


def animate(I, res=300, cmap="plasma", gif="", interval=75):
    """Animate the map as it rotates."""
    fig, ax = pl.subplots(1, figsize=(3, 3))
    I3D = I.reshape(-1, res, res)
    img = ax.imshow(I3D[0], origin="lower", interpolation="none", cmap=cmap,
                    extent=(-1, 1, -1, 1), animated=True,
                    vmin=np.nanmin(I3D), vmax=np.nanmax(I3D))
    ax.axis('off')

    def updatefig(i):
        img.set_array(I3D[i])
        return img,

    ani = animation.FuncAnimation(fig, updatefig, interval=interval,
                                  blit=True,
                                  frames=len(I3D))

    # Hack to return a gif embedded in HTML if we're
    # inside a Jupyter notebook
    try:
        if 'zmqshell' in str(type(get_ipython())):
            # We're inside a notebook!
            from IPython.display import HTML
            if gif == "":
                if not os.path.exists("_starry"):
                    os.mkdir("_starry")
                gif = os.path.join("_starry", str(id(ani)))
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

def show_spectral(I, res=300, cmap="plasma", gif="", interval=75):
    """Show the specific intensity I(x, y) on a grid."""
    fig, ax = pl.subplots(1, figsize=(3, 3))
    I3D = I.reshape(res, res, -1)
    frames = I3D.shape[2]
    img = ax.imshow(I3D[:, :, 0], origin="lower", 
                    interpolation="none", cmap=cmap,
                    extent=(-1, 1, -1, 1), animated=True,
                    vmin=np.nanmin(I3D), vmax=np.nanmax(I3D))
    ax.axis('off')
    if frames > 1:
        ax.set_title("%02d/%02d" % (1, frames))

    def updatefig(i):
        img.set_array(I3D[:, :, i])
        if frames > 1:
            ax.set_title("%02d/%02d" % (i + 1, frames))
        return img, ax

    # Set the interval automatically
    interval = int(75 * 50.0 / frames)
    if (interval < 50):
        interval = 50
    elif (interval > 500):
        interval = 500

    ani = animation.FuncAnimation(fig, updatefig, interval=interval,
                                  blit=False,
                                  frames=frames)

    # Hack to return a gif embedded in HTML if we're
    # inside a Jupyter notebook
    try:
        if 'zmqshell' in str(type(get_ipython())):
            # We're inside a notebook!
            from IPython.display import HTML
            if gif == "":
                if not os.path.exists("_starry"):
                    os.mkdir("_starry")
                gif = os.path.join("_starry", str(id(ani)))
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
    pl.show()
    pl.close()