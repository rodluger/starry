# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from ..utils import is_theano, to_tensor, vectorize, \
                    get_ortho_latitude_lines, get_ortho_longitude_lines
from ..ops import LimbDarkenedOp
import theano.tensor as tt


__all__ = ["LimbDarkenedBase"]


class LimbDarkenedBase(object):
    """
    .. automethod:: render
    .. automethod:: show
    .. automethod:: flux
    .. automethod:: __call__
    """

    def __init__(self, *args, **kwargs):
        super(LimbDarkenedBase, self).__init__(*args, **kwargs)
        self._flux_op = LimbDarkenedOp(self)

    @staticmethod
    def __descr__():
        return r"""
        A limb-darkened surface map with optional wavelength dependence.
        """

    def render(self, **kwargs):
        """
        Render the map on a grid and return the pixel intensities as a 
        two-dimensional array (with time as an optional third dimension).

        Kwargs:
            res (int): Map resolution, corresponding to the number of pixels \
                on a side. Default 300.
        """
        # Get kwargs
        res = kwargs.get("res", 300)

        # Render the map
        x, y = np.meshgrid(np.linspace(-1, 1, res), 
                           np.linspace(-1, 1, res))
        Z = np.array(self._intensity(np.sqrt(x ** 2 + y ** 2).flatten()))
        
        # Fix shape
        if self._spectral:
            Z = Z.reshape(res, res, self.nw)
            Z = np.moveaxis(Z, -1, 0)
        else:
            Z = Z.reshape(res, res)

        return np.squeeze(Z)

    def show(self, **kwargs):
        """
        Render and plot an image of the map; optionally display an animation.

        If running in a Jupyter Notebook, animations will be displayed
        in the notebook using javascript.
        Refer to the docstring of :py:meth:`render` for additional kwargs
        accepted by this method.

        Kwargs:
            Z (ndarray): The array of pixel intensities returned by a call \
                to :py:meth:`render`. Default :py:obj:`None`, in which case \
                this routine will call :py:meth:`render` with any additional \
                kwargs provided by the user.
            cmap: The colormap used for plotting (a string or a \
                :py:obj:`matplotlib` colormap object). Default "plasma".
            grid (bool): Overplot static grid lines? Default :py:obj:`True`.
            interval (int): Interval in ms between frames (animated maps only). \
                Default 75.
            mp4 (str): Name of the mp4 file to save the animation to \
                (animated maps only). Default :py:obj:`None`.
            kwargs: Any additional kwargs accepted by :py:meth:`render`.

        """
        # Get kwargs
        Z = kwargs.get("Z", None)
        cmap = kwargs.get("cmap", "plasma")
        grid = kwargs.get("grid", True)

        # Render the map
        if Z is None:
            Z = self.render(**kwargs)
        if len(Z.shape) == 3:
            nframes = Z.shape[0]
            animated = True
        else:
            nframes = 1
            Z = [Z]
            animated = False

        # Set up the plot
        fig, ax = plt.subplots(1, figsize=(3, 3))
        ax.axis('off')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        extent = (-1, 1, -1, 1)

        # Grid lines
        if grid:
            x = np.linspace(-1, 1, 10000)
            y = np.sqrt(1 - x ** 2)
            ax.plot(x, y, 'k-', alpha=1, lw=1)
            ax.plot(x, -y, 'k-', alpha=1, lw=1)
            for x, y in get_ortho_latitude_lines() + get_ortho_longitude_lines():
                ax.plot(x, y, 'k-', lw=0.5, alpha=0.5, zorder=100)

        # Plot the first frame of the image
        img = ax.imshow(Z[0], origin="lower", 
                        extent=extent, cmap=cmap,
                        interpolation="none",
                        vmin=np.nanmin(Z), vmax=np.nanmax(Z), 
                        animated=animated)
                
        # Display or save the image / animation
        if animated:
            interval = kwargs.pop("interval", 75)
            mp4 = kwargs.pop("mp4", None)
            
            def updatefig(i):
                img.set_array(Z[i])
                return img,

            ani = FuncAnimation(fig, updatefig, interval=interval,
                                blit=False, frames=len(Z))

            # Business as usual
            if (mp4 is not None) and (mp4 != ""):
                if mp4.endswith(".mp4"):
                    mp4 = mp4[:-4]
                ani.save('%s.mp4' % mp4, writer='ffmpeg')
                plt.close()
            else:
                try:
                    if 'zmqshell' in str(type(get_ipython())):
                        plt.close()
                        display(HTML(ani.to_jshtml()))
                    else:
                        raise NameError("")
                except NameError:
                    plt.show()
                    plt.close()
        else:
            plt.show()

    def flux(self, **kwargs):
        """
        Compute the flux visible from the map.

        Kwargs:
            b: The impact parameter of the occultor. Default 0.
            zo: The position of the occultor along \
                the line of sight. Default 1.0 (on the side closest to \
                the observer).
            ro: The radius of the occultor in units \
                of this body's radius. Default 0 (no occultation).
        
        Additional kwargs accepted by this method:
            u: The vector of limb darkening coefficients. Default \
                is the map's current limb darkening vector.
            orbit: And :py:obj:`exoplanet.orbits.KeplerianOrbit` instance. \
                This will override the :py:obj:`b` and :py:obj:`zo` keywords \
                above as long as a time vector :py:obj:`t` is also provided \
                (see below). Default :py:obj:`None`.
            t: A vector of times at which to evaluate the orbit. Default :py:obj:`None`.

        Returns:
            A vector of fluxes.
        """
        # Get the orbital coords
        orbit = kwargs.get("orbit", None)
        t = kwargs.get("t", None)
        if orbit is not None and t is not None:
            coords = orbit.get_relative_position(t)
            xo = coords[0] / orbit.r_star
            yo = coords[1] / orbit.r_star
            b = tt.sqrt(xo * xo + yo * yo)
            # Note that `exoplanet` uses a slightly different coord system!
            zo = -coords[2] / orbit.r_star
        else:
            b = kwargs.get("b", 0.0)
            zo = kwargs.get("zo", 1.0)
        ro = kwargs.get("ro", 0.0)
        u = kwargs.get("u", None)

        # Figure out if this is a Theano Op call
        if is_theano(u, b, zo, ro):
            if u is None:
                if self.udeg == 0:
                    u = []
                else:
                    if self._spectral:
                        u = self[1:, :]
                    else:
                        u = self[1:]
            u, b, zo, ro = to_tensor(u, b, zo, ro)
            b, zo, ro = vectorize(b, zo, ro)
            return self._flux_op(u, b, zo, ro)
        else:
            if u is not None:
                if self._spectral:
                    self[1:, :] = u
                else:
                    self[1:] = u
            return np.squeeze(self._flux(*vectorize(b, zo, ro)))

    def __call__(self, **kwargs):
        """
        Return the intensity of the map at a point or on a grid of surface points.

        Kwargs:
            b (float or ndarray): The impact parameter at the evaluation point.

        Returns:
            A vector of intensities at the corresponding surface point(s).

        """
        b = np.atleast_1d(kwargs.get("b", 0.0))
        return np.squeeze(self._intensity(b))