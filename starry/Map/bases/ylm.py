# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ...extensions import RAxisAngle
from ..sht import image2map, healpix2map, array2map
from ..utils import is_theano, to_tensor, vectorize, \
                    get_ortho_latitude_lines, get_ortho_longitude_lines
from ..ops import YlmXOp
from IPython.display import HTML
from scipy.optimize import minimize
import theano.tensor as tt


__all__ = ["YlmBase"]


class YlmBase(object):
    """
    .. automethod:: render
    .. automethod:: show
    .. automethod:: flux
    .. automethod:: __call__
    .. automethod:: load
    .. automethod:: align
    .. automethod:: max
    .. automethod:: min
    .. automethod:: is_physical
    """

    def __init__(self, *args, **kwargs):
        super(YlmBase, self).__init__(*args, **kwargs)
        self._X_op = YlmXOp(self)

    @staticmethod
    def __descr__():
        return r"""
        A basic :py:mod:`starry` surface map. The map is described
        as an expansion in spherical harmonics, with optional arbitrary
        order limb darkening and an optional multiplicative spherical
        harmonic filter. Support for wavelength-dependent and time-dependent
        maps is included, as well as flux and intensity calculation in
        reflected light.

        .. note:: Map instances are normalized such that the
            **average disk-integrated intensity is equal to unity**. The
            total luminosity over all :math:`4\pi` steradians is therefore
            :math:`4`. This normalization
            is particularly convenient for constant or purely limb-darkened
            maps, whose disk-integrated intensity is always equal to unity.
        """

    def render(self, **kwargs):
        r"""
        Render the map on a grid and return the pixel intensities as a 
        two-dimensional array (with time as an optional third dimension).

        Kwargs:
            theta (float): Angle of rotation of the map in degrees. Default 0.
            res (int): Map resolution, corresponding to the number of pixels \
                on a side (for the orthographic projection) or the number of \
                pixels in latitude (for the rectangular projection; the number \
                of pixels in longitude is twice this value). Default 300.
            projection (str): One of "orthographic" or "rectangular". The former \
                results in a map of the disk as seen on the plane of the sky, \
                padded by :py:obj:`NaN` outside of the disk. The latter results \
                in an equirectangular (geographic, equidistant cylindrical) \
                view of the entire surface of the map in latitude-longitude space. \
                Default "orthographic".
        
        Kwargs (temporal maps):
            t (float or ndarray): The time(s) at which to evaluate the map. \
                Default 0.
        
        Kwargs (reflected light maps):
            source (ndarray): A unit vector corresponding to the direction to the \
                light source. This may optionally be a vector of unit vectors. \
                Default :math:`-\hat{x}`.
        """
        # Get kwargs
        res = kwargs.get("res", 300)
        projection = kwargs.get("projection", "ortho")
        theta = kwargs.get("theta", 0.0)
        if projection.lower().startswith("rect"):
            projection = "rect"
            nframes = 1
            model_kwargs = dict()
        elif projection.lower().startswith("ortho"):
            projection = "ortho"
            if hasattr(theta, "__len__"):
                nframes = len(theta)
            else:
                nframes = 1
            model_kwargs = dict(theta=theta)
        else:
            raise ValueError("Invalid projection. Allowed projections are" +
                             "`rectangular` and `orthographic` (default).")

        # Are we modeling time variability?
        if self._temporal:
            t = kwargs.pop("t", 0.0)
            if hasattr(t, "__len__"):
                nframes = max(nframes, len(t))
            model_kwargs["t"] = t

        # Are we modeling reflected light?
        if self._reflected:
            source = kwargs.pop("source", 
                                [[-1.0, 0.0, 0.0] for n in range(nframes)])
            if source is None:
                # If explicitly set to `None`, re-run this
                # function on an *emitted* light map!
                from .. import Map
                if self._temporal:
                    map = Map(ydeg=self.ydeg, udeg=self.udeg, 
                              fdeg=self.fdeg, multi=self.multi, nt=self.nt)
                    map[:, :, :] = self[:, :, :]
                    if (self.udeg):
                        map[:] = self[:]
                    if (self.fdeg):
                        map.filter[:, :] = self.filter[:, :]
                    map.axis = self.axis
                    return map.render(theta=theta, res=res, 
                                      projection=projection, t=t)
                elif self._spectral:
                    map = Map(ydeg=self.ydeg, udeg=self.udeg, 
                              fdeg=self.fdeg, multi=self.multi, nw=self.nw)
                    map[:, :, :] = self[:, :, :]
                    if (self.udeg):
                        map[:] = self[:]
                    if (self.fdeg):
                        map.filter[:, :] = self.filter[:, :]
                    map.axis = self.axis
                    return map.render(theta=theta, res=res, 
                                      projection=projection)
                else:
                    map = Map(ydeg=self.ydeg, udeg=self.udeg, 
                              fdeg=self.fdeg, multi=self.multi)
                    map[:, :] = self[:, :]
                    if (self.udeg):
                        map[:] = self[:]
                    if (self.fdeg):
                        map.filter[:, :] = self.filter[:, :]
                    map.axis = self.axis
                    return map.render(theta=theta, res=res, 
                                      projection=projection)
            else:
                source = np.ascontiguousarray(source)
                if len(source.shape) == 2:
                    nframes = max(nframes, len(source))
                model_kwargs["source"] = source

        # Are we doing wavelength dependence?
        if self._spectral:
            assert nframes == 1, "Spectral map rotation cannot be animated."

        if projection == "rect":

            # Disable limb darkening & filter
            if self.udeg:
                u_copy = np.array(self[1:])
                self[1:] = 0
            if self.fdeg:
                f_copy = np.array(self.filter[:, :])
                self.filter[:, :] = 0

            # Generate the lat/lon grid for one hemisphere
            lon = np.linspace(-np.pi, np.pi, res)
            lat = np.linspace(1e-3, np.pi / 2, res // 2)
            lon, lat = np.meshgrid(lon, lat)
            x = np.sin(np.pi / 2 - lat) * np.cos(lon - np.pi / 2)
            y = np.sin(np.pi / 2 - lat) * np.sin(lon - np.pi / 2)

            # Rotate so we're looking down the north pole
            map_axis = np.array(self.axis)
            alpha = np.arccos(np.dot(map_axis, [0, 0, 1])) * 180 / np.pi
            u = np.cross(map_axis, [0, 0, 1])
            self.axis = u
            self.rotate(alpha)

            # We need to rotate the light source as well
            if self._reflected:
                R = RAxisAngle(u, alpha)
                source = np.atleast_2d(model_kwargs["source"])
                for i in range(len(source)):
                    source[i] = np.dot(R, source[i])
                model_kwargs["source"] = source

            # Compute the northern hemisphere map
            self.axis = [0, 0, 1]
            Z_north = np.array(self(x=x, y=y, **model_kwargs))
            if self._spectral:
                Z_north = Z_north.reshape(res // 2, res, self.nw)
                Z_north = np.moveaxis(Z_north, -1, 0)
            else:
                Z_north = Z_north.reshape(nframes, res // 2, res)

            # Flip the planet around
            self.axis = [1, 0, 0]
            self.rotate(180)

            # We need to rotate the light source as well
            if self._reflected:
                R = RAxisAngle([1, 0, 0], 180)
                source = np.atleast_2d(model_kwargs["source"])
                for i in range(len(source)):
                    source[i] = np.dot(R, source[i])
                model_kwargs["source"] = source
            
            # Compute the southern hemisphere map
            self.axis = [0, 0, -1]
            Z_south = np.array(self(x=-x, y=-y, **model_kwargs))
            if self._spectral:
                Z_south = Z_south.reshape(res // 2, res, self.nw)
                Z_south = np.moveaxis(Z_south, -1, 0)
            else:
                Z_south = Z_south.reshape(nframes, res // 2, res)
            Z_south = np.flip(Z_south, axis=(1, 2))

            # Join them
            Z = np.concatenate((Z_south, Z_north), axis=1)

            # Undo all the rotations
            self.axis = [1, 0, 0]
            self.rotate(-180)
            self.axis = u
            self.rotate(-alpha)
            self.axis = map_axis

            # Re-enable limb darkening & filter
            if self.udeg:
                self[1:] = u_copy
            if self.fdeg:
                self.filter[:, :] = f_copy

        else:

            # Create a grid of X and Y and construct the linear model
            x, y = np.meshgrid(np.linspace(-1, 1, res), 
                               np.linspace(-1, 1, res))
            Z = np.array(self(x=x, y=y, **model_kwargs))
            if self._spectral:
                Z = Z.reshape(res, res, self.nw)
                Z = np.moveaxis(Z, -1, 0)
            else:
                Z = Z.reshape(nframes, res, res)

        return np.squeeze(Z)

    def show(self, **kwargs):
        """
        Render and plot an image of the map; optionally display an animation.

        If running in a Jupyter Notebook, animations will be displayed
        in the notebook using javascript.
        Refer to the docstring of :py:meth:`render` for additional kwargs
        accepted by this method.

        Args:
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
        projection = kwargs.get("projection", "ortho")
        grid = kwargs.get("grid", True)

        # Render the map
        if Z is None:
            Z = self.render(**kwargs)
        if len(Z.shape) == 3:
            nframes = Z.shape[0]
        else:
            nframes = 1
            Z = [Z]

        # Are we doing wavelength dependence?
        if self._spectral:
            animated = True
        else:
            animated = (nframes > 1)

        if projection == "rect":
            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(7, 3.75))
            extent = (-180, 180, -90, 90)

            if grid:
                latlines = np.linspace(-90, 90, 7)[1:-1]
                lonlines = np.linspace(-180, 180, 13)
                for lat in latlines:
                    ax.axhline(lat, color="k", lw=0.5, alpha=0.5, zorder=100)
                for lon in lonlines:
                    ax.axvline(lon, color="k", lw=0.5, alpha=0.5, zorder=100)
            ax.set_xticks(lonlines)
            ax.set_yticks(latlines)
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")

        else:
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
                lat_lines = get_ortho_latitude_lines(inc=self.inc, obl=self.obl)
                lon_lines = get_ortho_longitude_lines(inc=self.inc, obl=self.obl)
                for x, y in lat_lines + lon_lines:
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

    def X(self, **kwargs):
        r"""
        Compute the flux design matrix.

        Kwargs:
            theta (float or ndarray): Angle of rotation. Default 0.
            xo (float or ndarray): The :py:obj:`x` position of the \
                occultor (if any). Default 0.
            yo (float or ndarray): The :py:obj:`y` position of the \
                occultor (if any). Default 0.
            zo (float or ndarray): The :py:obj:`z` position of the \
                occultor (if any). Default 1.0 (on the side closest to \
                the observer).
            ro (float): The radius of the occultor in units of this \
                body's radius. Default 0 (no occultation).
        
        Kwargs (temporal maps or if :py:obj:`orbit` is provided):
            t: Time at which to evaluate the map and/or orbit. Default 0. 

        Kwargs (reflected light maps):
            source: The source position, a unit vector or a
                vector of unit vectors. Default :math:`-\hat{x} = (-1, 0, 0)`.

        Additional kwargs accepted by this method:
            u: The vector of limb darkening coefficients. Default \
                is the map's current limb darkening vector.
            f: The vector of filter coefficients. Default \
                is the map's current filter vector.
            inc: The map inclination in degrees. Default is the map's current \
                inclination.
            obl: The map obliquity in degrees. Default is the map's current \
                obliquity. 
            orbit: And :py:obj:`exoplanet.orbits.KeplerianOrbit` instance. \
                This will override the :py:obj:`b` and :py:obj:`zo` keywords \
                above as long as a time vector :py:obj:`t` is also provided \
                (see above). Default :py:obj:`None`.

        Returns:
            The design matrix :py:obj:`X`.
        """
        # TODO!
        if self._spectral or self._temporal:
            raise NotImplementedError("Not yet implemented!")

        # Get the orbital coords
        orbit = kwargs.get("orbit", None)
        t = kwargs.get("t", None)
        if orbit is not None and t is not None:
            coords = orbit.get_relative_position(t)
            xo = coords[0] / orbit.r_star
            yo = coords[1] / orbit.r_star
            # Note that `exoplanet` uses a slightly different coord system!
            zo = -coords[2] / orbit.r_star
        else:
            xo = kwargs.get("xo", 0.0)
            yo = kwargs.get("yo", 0.0)
            zo = kwargs.get("zo", 1.0)
        theta = kwargs.get("theta", 0.0)
        ro = kwargs.get("ro", 0.0)

        # Additional kwargs
        u = kwargs.get("u", None)
        f = kwargs.get("f", None)
        inc = kwargs.get("inc", None)
        obl = kwargs.get("obl", None)

        # Figure out if this is a Theano Op call
        if is_theano(u, f, inc, obl, theta, xo, yo, zo, ro):
            if u is None:
                if self.udeg == 0:
                    u = []
                else:
                    u = self[1:]
            if f is None:
                if self.fdeg == 0:
                    f = []
                else:
                    f = self.f
            if inc is None:
                inc = self.inc
            if obl is None:
                obl = self.obl
            u, f, inc, obl, theta, xo, yo, zo, ro = \
                to_tensor(u, f, inc, obl, theta, xo, yo, zo, ro)
            theta, xo, yo, zo, ro = vectorize(theta, xo, yo, zo, ro)
            return self._X_op(u, f, inc, obl, theta, xo, yo, zo, ro)
        else:
            if u is not None:
                self[1:] = u
            if f is not None:
                self._set_filter((slice(None), slice(None)), f)
            if inc is not None:
                self.inc = inc
            if obl is not None:
                self.obl = obl
            return np.squeeze(self._X(*vectorize(theta, xo, yo, zo, ro)))

    def flux(self, **kwargs):
        r"""
        Compute the flux.

        Kwargs:
            theta (float or ndarray): Angle of rotation. Default 0.
            xo (float or ndarray): The :py:obj:`x` position of the \
                occultor (if any). Default 0.
            yo (float or ndarray): The :py:obj:`y` position of the \
                occultor (if any). Default 0.
            zo (float or ndarray): The :py:obj:`z` position of the \
                occultor (if any). Default 1.0 (on the side closest to \
                the observer).
            ro (float): The radius of the occultor in units of this \
                body's radius. Default 0 (no occultation).
        
        Kwargs (temporal maps or if :py:obj:`orbit` is provided):
            t: Time at which to evaluate the map and/or orbit. Default 0. 

        Kwargs (reflected light maps):
            source: The source position, a unit vector or a
                vector of unit vectors. Default :math:`-\hat{x} = (-1, 0, 0)`.

        Additional kwargs accepted by this method:
            y: The vector of spherical harmonic coefficients. Default \
                is the map's current spherical harmonic vector.
            u: The vector of limb darkening coefficients. Default \
                is the map's current limb darkening vector.
            f: The vector of filter coefficients. Default \
                is the map's current filter vector.
            inc: The map inclination in degrees. Default is the map's current \
                inclination.
            obl: The map obliquity in degrees. Default is the map's current \
                obliquity. 
            orbit: And :py:obj:`exoplanet.orbits.KeplerianOrbit` instance. \
                This will override the :py:obj:`b` and :py:obj:`zo` keywords \
                above as long as a time vector :py:obj:`t` is also provided \
                (see above). Default :py:obj:`None`.

        Returns:
            The flux timeseries.
        """
        # Ingest the map coefficients
        if self.ydeg:
            y = kwargs.get("y", None)
            if y is None:
                if self._scalar:
                    y = self[1:, :]
                else:
                    y = self[1:, :, :]
            elif not is_theano(y):
                if self._scalar:
                    self[1:, :] = y
                else:
                    self[1:, :, :] = y
        else:
            y = []
            
        # Compute the design matrix
        X = self.X(**kwargs)

        # Dot it into the map
        if is_theano(X):
            return tt.dot(X, tt.join(0, [1.0], y))
        else:
            return np.dot(X, np.append([1.0], y))

    def __call__(self, **kwargs):
        r"""
        Return the intensity of the map at a point or on a grid of surface points.

        Kwargs:
            theta (float or ndarray): Angle of rotation. Default 0.
            x (float or ndarray): The :py:obj:`x` position on the \
                surface. Default 0.
            y (float or ndarray): The :py:obj:`y` position on the \
                surface. Default 0.

        Kwargs (temporal maps):
            t: Time at which to evaluate. Default 0. 

        Kwargs (reflected light maps):
            source: The source position, a unit vector or a
                vector of unit vectors. Default :math:`-\hat{x} = (-1, 0, 0)`.

        Returns:
            A vector of intensities at the corresponding surface point(s).

        """
        theta = np.atleast_1d(kwargs.get("theta", 0.0))
        x = np.atleast_1d(kwargs.get("x", 0.0))
        y = np.atleast_1d(kwargs.get("y", 0.0))
        x, y = vectorize(x, y)
        if self._temporal:
            t = np.atleast_1d(kwargs.get("t", 0.0))
            t, theta = vectorize(t, theta)
            args = [t, theta, x, y]
        elif self._spectral:
            source = np.atleast_1d(kwargs.get("source", [-1.0, 0.0, 0.0]))
            if source.ndim == 1:
                source = [source for i in theta]
            elif theta.ndim == 1:
                theta = [theta for s in source]
            args = [theta, x, y, source]
        else:
            args = [theta, x, y]
        return np.squeeze(self._intensity(*args).reshape(len(theta), -1))

    def load(self, image, ydeg=None, healpix=False, col=None, **kwargs):
        """
        Load an image, array, or :py:obj:`healpix` map. 
        
        This routine uses
        various routines in :py:obj:`healpix` to compute the spherical
        harmonic expansion of the input image and sets the map's :py:attr:`y`
        coefficients accordingly.

        Args:
            image: A path to an image file, a two-dimensional :py:obj:`numpy` \
                array, or a :py:obj:`healpix` map array \
                (if :py:obj:`healpix = True`).
            ydeg (int): The degree of the spherical harmonic expansion of the \
                image. Default :py:obj:`None`, in which case the expansion is \
                performed up to :py:obj:`self.ydeg`.
            healpix (bool): Treat :py:obj:`image` as a :py:obj:`healpix` array? \
                Default :py:obj:`False`.
            col: The map column into which the image will be loaded. Can be \
                an :py:obj:`int`, :py:obj:`slice`, or :py:obj:`None`. \
                Default :py:obj:`None`, in which case the image is loaded into \
                the first map column. This option is ignored for scalar maps.
            sampling_factor (int): Oversampling factor when computing the \
                :py:obj:`healpix` representation of an input image or array. \
                Default 8. Increasing this number may improve the fidelity of \
                the expanded map, but the calculation will take longer.
            sigma (float): If not :py:obj:`None`, apply gaussian smoothing \
                with standard deviation :py:obj:`sigma` to smooth over \
                spurious ringing features. Smoothing is performed with \
                the :py:obj:`healpix.sphtfunc.smoothalm` method. \
                Default :py:obj:`None`.

        .. note:: Method not available for purely limb-darkened maps.
        
        """
        if col is None:
            col = 0

        # Check the degree
        if ydeg is None:
            ydeg = self.ydeg
        assert (ydeg <= self.ydeg) and (ydeg > 0), \
            "Invalid spherical harmonic degree."
        
        # Is this a file name?
        if type(image) is str:
            y = image2map(image, lmax=ydeg, **kwargs)
        # or is it an array?
        elif (type(image) is np.ndarray):
            if healpix:
                y = healpix2map(image, lmax=ydeg, **kwargs)
            else:
                y = array2map(image, lmax=ydeg, **kwargs)
        else:
            raise ValueError("Invalid `image` value.")
        
        # Ingest the coefficients
        if self._spectral or self._temporal:
            self[1:, :, :] = 0
            self[:ydeg + 1, :, col] = y
        else:
            self[1:, :] = 0
            self[:ydeg + 1, :] = y
    
    def align(self, source=None, dest=None):
        """
        Rotate the map to align the :py:obj:`source` point/axis with the
        :py:obj:`dest` point/axis.

        The standard way of rotating maps in :py:obj:`starry` is to
        provide the axis and angle of rotation, but this isn't always
        convenient. In some cases, it is easier to specify a source
        point/axis and a destination point/axis, and rotate the map such that the
        source aligns with the destination. This is particularly useful for
        changing map projections. For instance, to view the map pole-on,

        .. code-block:: python

            map.align(source=map.axis, dest=[0, 0, 1])

        This rotates the map axis to align with the z-axis, which points
        toward the observer.

        Args:
            source (ndarray): A unit vector describing the source position. \
                This point will be rotated onto :py:obj:`dest`. Default \
                is the current map axis.
            dest (ndarray): A unit vector describing the destination position. \
                The :py:obj:`source` point will be rotated onto this point. Default \
                is the current map axis.

        """
        if source is None:
            source = self.axis
        if dest is None:
            dest = self.axis
        self.rotate(axis=np.cross(source, dest), 
                    theta=np.arccos(np.dot(source, dest)) * 180 / np.pi)
    
    def _extremum(self, minimum=True):
        """
        Find a global extremum of the map.

        .. todo:: Speed this up with gradients and remove the overhead \
            of calling `linear_model` every time. Set up unit tests for \
            this method.
        """
        # Keep the minimizer on the unit disk
        cons = [{'type': 'ineq', 
                 'fun':  lambda x: 1 - x[0] ** 2 - x[1] ** 2}]
        
        # Start in the center
        x0 = [0, 0]

        # The objective function
        if minimum:
            def func(x, theta):
                # Beware the caching!
                return np.float64(self(theta=theta, x=x[0], y=x[1]))
        else:
            def func(x, theta):
                # Beware the caching!
                return -np.float64(self(theta=theta, x=x[0], y=x[1]))

        # Disable limb darkening & filter
        if self.udeg:
            u_copy = np.array(self[1:])
            self[1:] = 0
        if self.fdeg:
            f_copy = np.array(self.f)
            self._set_filter((slice(None), slice(None)), np.zeros(self.Nf))

        # Front side, then back side
        res_f = minimize(func, x0, args=(0))
        res_b = minimize(func, x0, args=(180))
        
        # Re-enable limb darkening & filter
        if self.udeg:
            self[1:] = u_copy
        if self.fdeg:
            self._set_filter((slice(None), slice(None)), f_copy)

        # Return the extremum
        if minimum:
            return min(res_f.fun, res_b.fun)
        else:
            return -min(res_f.fun, res_b.fun)

    def min(self):
        """
        Return the global minimum of the intensity.

        This routine uses `scipy.optimize.minimize` to attempt to find
        the global minimum. Note that both the limb darkening and the
        multiplicative filter are disabled for this method.

        .. warning:: This routine is not yet optimized and has not been \
            fully tested. It may be \
            unnecessarily slow and may not always find the global \
            minimum. This will be fixed in an upcoming version of the code.
        """
        return self._extremum(True)
    
    def max(self):
        """
        Return the global maximum of the intensity.

        This routine uses `scipy.optimize.minimize` to attempt to find
        the global maximum. Note that both the limb darkening and the
        multiplicative filter are disabled for this method.

        .. warning:: This routine is not yet optimized and has not been \
            fully tested. It may be \
            unnecessarily slow and may not always find the global \
            maximum. This will be fixed in an upcoming version of the code.
        """
        return self._extremum(False)
    
    def is_physical(self):
        """
        Returns :py:obj:`True` if the map intensity is non-negative
        everywhere. 
        
        This routine uses `scipy.optimize.minimize` to attempt to find
        the global minimum. Note that both the limb darkening and the
        multiplicative filter are ignored.

        .. warning:: This routine is not yet optimized and has not been \
            fully tested. It may be \
            unnecessarily slow and may not always find the global \
            minimum. This will be fixed in an upcoming version of the code.
        """
        return self.min() >= 0