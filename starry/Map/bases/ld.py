# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from ..utils import is_theano, to_tensor, vectorize, \
                    get_ortho_latitude_lines, get_ortho_longitude_lines
from ..ops import LimbDarkenedOp
import theano.tensor as tt
import warnings


__all__ = ["LimbDarkenedBase"]


class LimbDarkenedBase(object):
    """
    .. automethod:: render
    .. automethod:: show
    .. automethod:: flux
    .. automethod:: __call__
    """

    __descr__ = \
    """
    A limb-darkened surface map with optional wavelength dependence.
    Instantiate by calling
    
    .. code-block:: python

        starry.Map(udeg=udeg, **kwargs)

    with ``udeg > 0``. Note that limb-darkened maps cannot
    (currently) have temporal dependence and must be in emitted
    light only, although users can instantiate a :py:class:`SphericalHarmonicMap`
    in reflected light and add limb darkening.
    """

    def __init__(self, *args, **kwargs):
        super(LimbDarkenedBase, self).__init__(*args, **kwargs)
        self._flux_op = LimbDarkenedOp(self)

    def render(self, **kwargs):
        """
        Render the map on a grid and return the pixel intensities as a 
        two-dimensional array (with time as an optional third dimension).

        Keyword Arguments:
            res (int): Map resolution, corresponding to the number of pixels
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

        Keyword Arguments:
            Z (ndarray): The array of pixel intensities returned by a call 
                to :py:meth:`render`. Default ``None``, in which case 
                this routine will call :py:meth:`render` with any additional 
                kwargs provided by the user.
            cmap: The colormap used for plotting (a string or a 
                ``matplotlib`` colormap object). Default "plasma".
            grid (bool): Overplot static grid lines? Default ``True``.
            interval (int): Interval in ms between frames (animated maps only). 
                Default 75.
            mp4 (str): Name of the mp4 file to save the animation to 
                (animated maps only). Default ``None``.
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

        Keyword Arguments:
            b: The impact parameter of the occultor. Default 0.
            zo: The position of the occultor along 
                the line of sight. Default 1.0 (on the side closest to 
                the observer).
            ro: The radius of the occultor in units 
                of this body's radius. Default 0 (no occultation).
        
        The following arguments are also accepted:

        Keyword Arguments:
            u: The vector of limb darkening coefficients. Default 
                is the map's current limb darkening vector.
            orbit: And ``exoplanet.orbits.KeplerianOrbit`` instance. 
                This will override the ``b`` and ``zo`` keywords 
                above as long as a time vector ``t`` is also provided 
                (see below). Default ``None``.
            texp: The exposure time of each observation. 
                This can be a scalar or a vector/tensor with the same shape as ``t``. 
                If ``texp`` is provided, ``t`` is assumed to indicate the 
                timestamp at the *middle* of an exposure of length ``texp``. 
                Only applies if ``orbit`` is provided. Default ``None``.
            use_in_transit (bool): If ``True``, the model will only 
                be evaluated for the data points expected to be in transit 
                as computed using the ``in_transit`` method on ``orbit``. 
                Only applies if ``orbit`` is provided. Default ``True``.
            oversample (int): The number of function evaluations to 
                use when numerically integrating the exposure time. 
                Only applies if ``orbit`` is provided. Default ``7``.
            order (int): The order of the numerical integration 
                scheme. This must be one of the following: ``0`` for a 
                centered Riemann sum (equivalent to the "resampling" procedure 
                suggested by Kipping 2010), ``1`` for the trapezoid rule, or 
                ``2`` for Simpson's rule. 
                Only applies if ``orbit`` is provided. Default ``0``.
            t: A vector of times at which to evaluate the orbit. Default ``None``.

        Returns:
            A vector of fluxes or a ``Theano`` Op corresponding to the flux 
            computation.
        """
        # Ingest kwargs
        u = kwargs.pop("u", None)
        b = kwargs.pop("b", 0.0)
        zo = kwargs.pop("zo", kwargs.pop("z", 1.0)) # Lenient! Can provide `z`
        ro = kwargs.pop("ro", kwargs.pop("r", 0.0)) # Lenient! Can provide `r`
        orbit = kwargs.pop("orbit", None)
        t = kwargs.pop("t", None)
        use_in_transit = kwargs.pop("use_in_transit", True)
        texp = kwargs.pop("texp", None)
        oversample = kwargs.pop("oversample", 7)
        order = kwargs.pop("order", 0)

        # Raise warnings for some combinations
        if (orbit is not None) and (t is None):
            raise ValueError("Please provide a set of times `t`.")
        if (orbit is None or t is None):
            use_in_transit = False
            if texp is not None:
                warnings.warn("Exposure time integration enabled only " +
                              "when an `orbit` instance is provided.") 
        for kwarg in kwargs.keys():
            warnings.warn("Unrecognized kwarg: %s. Ignoring..." % kwarg)

        # Figure out if this is a Theano Op call
        if (orbit is not None and t is not None) or is_theano(u, b, zo, ro):

            # Limb darkening coeffs
            if u is None:
                if self.udeg == 0:
                    u = []
                else:
                    if self._spectral:
                        u = self[1:, :]
                    else:
                        u = self[1:]
            u = to_tensor(u)

            # Figure out the coords from the orbit
            if orbit is not None and t is not None:
                
                # Tensorize the time array
                t = to_tensor(t)
                
                # Only compute during transit?
                if use_in_transit:
                    transit_model = tt.ones_like(t)
                    transit_inds = orbit.in_transit(t, r=ro, texp=texp)
                    t = t[transit_inds]

                # Exposure time grid
                if texp is None:
                    
                    # Easy
                    tgrid = t
                
                else:
                    
                    # From DFM's exoplanet
                    texp = to_tensor(texp)
                    oversample = int(oversample)
                    oversample += 1 - oversample % 2
                    stencil = np.ones(oversample)

                    # Construct the exposure time integration stencil
                    if order == 0:
                        dt = np.linspace(-0.5, 0.5, 2*oversample+1)[1:-1:2]
                    elif order == 1:
                        dt = np.linspace(-0.5, 0.5, oversample)
                        stencil[1:-1] = 2
                    elif order == 2:
                        dt = np.linspace(-0.5, 0.5, oversample)
                        stencil[1:-1:2] = 4
                        stencil[2:-1:2] = 2
                    else:
                        raise ValueError("Keyword `order` must be <= 2.")
                    stencil /= np.sum(stencil)

                    if texp.ndim == 0:
                        dt = texp * dt
                    else:
                        if use_in_transit:
                            dt = tt.shape_padright(texp[transit_inds]) * dt
                        else:
                            dt = tt.shape_padright(texp) * dt

                    tgrid = tt.shape_padright(t) + dt
                    tgrid = tt.reshape(tgrid, [-1])

                # Compute coords
                coords = orbit.get_relative_position(tgrid)
                xo = coords[0] / orbit.r_star
                yo = coords[1] / orbit.r_star
                b = tt.sqrt(xo * xo + yo * yo)
                # Note that `exoplanet` uses a slightly different coord system!
                zo = -coords[2] / orbit.r_star

            else:

                # Tensorize & vectorize
                b, zo = vectorize(b, zo)

            # Compute the light curve
            lc = self._flux_op(u, b, zo, ro)

            # Integrate it
            if texp is not None:
                stencil = tt.shape_padleft(stencil, t.ndim)
                lc = tt.squeeze(tt.sum(stencil * tt.reshape(lc, 
                                [t.shape[0], oversample]), axis=t.ndim))

            # Return the full model
            if use_in_transit:
                transit_model = tt.set_subtensor(transit_model[transit_inds], lc)
                return transit_model
            else:
                return lc

        else:

            # No Theano nonsense!
            if u is not None:
                if self._spectral:
                    self[1:, :] = u
                else:
                    self[1:] = u
            return np.squeeze(self._flux(*vectorize(b, zo), ro))

    def __call__(self, **kwargs):
        """
        Return the intensity of the map at a point or on a grid of surface points.

        Keyword Arguments:
            b (float or ndarray): The impact parameter at the evaluation point.

        Returns:
            A vector of intensities at the corresponding surface point(s).

        """
        b = np.atleast_1d(kwargs.get("b", 0.0))
        return np.squeeze(self._intensity(b))