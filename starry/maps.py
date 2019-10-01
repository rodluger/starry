# -*- coding: utf-8 -*-

# TODO:
# - L normalization: is the integral of I equal to L?
# - Reflected light maps: what is L? Make it prop to 1/r^2
# - Is sys.secondaries[i] a ptr as before? Check.
# - Check how map.load() normalizes things.
# - Reflected light: get rid of `source`; just use `xo`, `yo`, `zo`
# - MAP Op

from . import config
from .ops import (
    Ops,
    OpsReflected,
    OpsRV,
    vectorize,
    atleast_2d,
    get_projection,
    is_theano,
    reshape,
    STARRY_RECTANGULAR_PROJECTION,
    STARRY_ORTHOGRAPHIC_PROJECTION,
)
from .indices import integers, get_ylm_inds, get_ul_inds, get_ylmw_inds
from .utils import get_ortho_latitude_lines, get_ortho_longitude_lines
from .sht import image2map, healpix2map, array2map
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from warnings import warn
from astropy import units
import os


__all__ = ["Map", "MapBase", "YlmBase", "RVBase", "ReflectedBase"]


class Luminosity(object):
    def __get__(self, instance, owner):
        return instance._L

    def __set__(self, instance, value):
        instance._L = instance.cast(np.ones(instance.nw) * value)


class MapBase(object):
    """The base class for all `starry` maps."""

    pass


class YlmBase(object):
    """The default ``starry`` map class.

    This class handles light curves and phase curves of objects in 
    emitted light. It can be instantiated by calling :py:func:`starry.Map` with 
    both ``rv`` and ``reflected`` set to False.
    """

    _ops_class_ = Ops
    L = Luminosity()

    def __init__(self, ydeg, udeg, fdeg, drorder, nw, quiet=False, **kwargs):
        """

        """
        # Instantiate the Theano ops class
        self.quiet = quiet
        self.ops = self._ops_class_(
            ydeg, udeg, fdeg, drorder, nw, quiet=quiet, **kwargs
        )
        self.cast = self.ops.cast

        # Dimensions
        self._ydeg = ydeg
        self._Ny = (ydeg + 1) ** 2
        self._udeg = udeg
        self._Nu = udeg + 1
        self._fdeg = fdeg
        self._Nf = (fdeg + 1) ** 2
        self._deg = ydeg + udeg + fdeg
        self._N = (ydeg + udeg + fdeg + 1) ** 2
        self._nw = nw

        # Units
        self.angle_unit = kwargs.pop("angle_unit", units.degree)

        # Initialize
        self.reset(**kwargs)

    @property
    def angle_unit(self):
        """An ``astropy.units`` unit defining the angle metric for this map."""
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, value):
        assert value.physical_type == "angle"
        self._angle_unit = value
        self._angle_factor = value.in_units(units.radian)

    @property
    def ydeg(self):
        """Spherical harmonic degree of the map. *Read-only*"""
        return self._ydeg

    @property
    def Ny(self):
        r"""Number of spherical harmonic coefficients. *Read-only*

        This is equal to :math:`(y_\mathrm{deg} + 1)^2`.
        """
        return self._Ny

    @property
    def udeg(self):
        """Limb darkening degree. *Read-only*"""
        return self._udeg

    @property
    def Nu(self):
        r"""Number of limb darkening coefficients, including :math:`u_0`. *Read-only*
        
        This is equal to :math:`u_\mathrm{deg} + 1`.
        """
        return self._Nu

    @property
    def fdeg(self):
        """Degree of the multiplicative filter. *Read-only*"""
        return self._fdeg

    @property
    def Nf(self):
        r"""Number of spherical harmonic coefficients in the filter. *Read-only*

        This is equal to :math:`(f_\mathrm{deg} + 1)^2`.
        """
        return self._Nf

    @property
    def deg(self):
        r"""Total degree of the map. *Read-only*
        
        This is equal to :math:`y_\mathrm{deg} + u_\mathrm{deg} + f_\mathrm{deg}`.
        """
        return self._deg

    @property
    def N(self):
        """Total number of map coefficients. *Read-only*
        
        This is equal to :math:`N_\mathrm{y} + N_\mathrm{u} + N_\mathrm{f}`.
        """
        return self._N

    @property
    def nw(self):
        """Number of wavelength bins. *Read-only*"""
        return self._nw

    @property
    def y(self):
        """The spherical harmonic coefficient vector. *Read-only*
        
        To set this vector, index the map directly using two indices:
        ``map[l, m] = ...`` where ``l`` is the spherical harmonic degree and 
        ``m`` is the spherical harmonic order. These may be integers or 
        arrays of integers. Slice notation may also be used.
        """
        return self._y

    @property
    def u(self):
        """The vector of limb darkening coefficients. *Read-only*
        
        To set this vector, index the map directly using one index:
        ``map[n] = ...`` where ``n`` is the degree of the limb darkening
        coefficient. This may be an integer or an array of integers.
        Slice notation may also be used.
        """
        return self._u

    @property
    def inc(self):
        """The inclination of the rotation axis in units of :py:attr:`angle_unit`."""
        return self._inc / self._angle_factor

    @inc.setter
    def inc(self, value):
        self._inc = self.cast(value) * self._angle_factor

    @property
    def obl(self):
        """The obliquity of the rotation axis in units of :py:attr:`angle_unit`."""
        return self._obl / self._angle_factor

    @obl.setter
    def obl(self, value):
        self._obl = self.cast(value) * self._angle_factor

    @property
    def alpha(self):
        """The rotational shear coefficient, a number in the range ``[0, 1]``.
        
        The parameter :math:`\\alpha` is used to model linear differential
        rotation. The angular velocity at a given latitude :math:`\\theta`
        is

        :math:`\\omega = \\omega_{eq}(1 - \\alpha \\sin^2\\theta)`

        where :math:`\\omega_{eq}` is the equatorial angular velocity of
        the object.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = self.cast(value)

    def __getitem__(self, idx):
        if isinstance(idx, integers) or isinstance(idx, slice):
            # User is accessing a limb darkening index
            inds = get_ul_inds(self.udeg, idx)
            return self._u[inds]
        elif isinstance(idx, tuple) and len(idx) == 2 and self.nw is None:
            # User is accessing a Ylm index
            inds = get_ylm_inds(self.ydeg, idx[0], idx[1])
            return self._y[inds]
        elif isinstance(idx, tuple) and len(idx) == 3 and self.nw:
            # User is accessing a Ylmw index
            inds = get_ylmw_inds(self.ydeg, self.nw, idx[0], idx[1], idx[2])
            return self._y[inds]
        else:
            raise ValueError("Invalid map index.")

    def __setitem__(self, idx, val):
        if isinstance(idx, integers) or isinstance(idx, slice):
            # User is accessing a limb darkening index
            inds = get_ul_inds(self.udeg, idx)
            if 0 in inds:
                raise ValueError("The u_0 coefficient cannot be set.")
            if config.lazy:
                self._u = self.ops.set_map_vector(self._u, inds, val)
            else:
                self._u[inds] = val
        elif isinstance(idx, tuple) and len(idx) == 2 and self.nw is None:
            # User is accessing a Ylm index
            inds = get_ylm_inds(self.ydeg, idx[0], idx[1])
            if 0 in inds:
                raise ValueError("The Y_{0,0} coefficient cannot be set.")
            if config.lazy:
                self._y = self.ops.set_map_vector(self._y, inds, val)
            else:
                self._y[inds] = val
        elif isinstance(idx, tuple) and len(idx) == 3 and self.nw:
            # User is accessing a Ylmw index
            inds = get_ylmw_inds(self.ydeg, self.nw, idx[0], idx[1], idx[2])
            if 0 in inds[0]:
                raise ValueError("The Y_{0,0} coefficients cannot be set.")
            if config.lazy:
                self._y = self.ops.set_map_vector(self._y, inds, val)
            else:
                old_shape = self._y[inds].shape
                new_shape = np.atleast_2d(val).shape
                if old_shape == new_shape:
                    self._y[inds] = val
                elif old_shape == new_shape[::-1]:
                    self._y[inds] = np.atleast_2d(val).T
                else:
                    self._y[inds] = val
        else:
            raise ValueError("Invalid map index.")

    def _check_kwargs(self, method, kwargs):
        if not self.quiet:
            for key in kwargs.keys():
                message = "Invalid keyword `{0}` in call to `{1}()`. Ignoring."
                message = message.format(key, method)
                warn(message)

    def _get_orbit(self, kwargs):
        xo = kwargs.pop("xo", 0.0)
        yo = kwargs.pop("yo", 0.0)
        zo = kwargs.pop("zo", 1.0)
        ro = kwargs.pop("ro", 0.0)
        theta = kwargs.pop("theta", 0.0)
        theta, xo, yo, zo = vectorize(theta, xo, yo, zo)
        theta, xo, yo, zo, ro = self.cast(theta, xo, yo, zo, ro)
        theta *= self._angle_factor
        return theta, xo, yo, zo, ro

    def reset(self, **kwargs):
        """Reset all map coefficients and attributes.
        
        .. note:: 
            Does not reset custom unit settings.
        
        """
        if self.nw is None:
            y = np.zeros(self.Ny)
            y[0] = 1.0
        else:
            y = np.zeros((self.Ny, self.nw))
            y[0, :] = 1.0
        self._y = self.cast(y)

        u = np.zeros(self.Nu)
        u[0] = -1.0
        self._u = self.cast(u)

        f = np.zeros(self.Nf)
        f[0] = np.pi
        self._f = self.cast(f)

        self._L = self.cast(kwargs.pop("L", np.ones(self.nw)))

        if kwargs.get("inc", None) is not None:
            self.inc = kwargs.pop("inc")
        else:
            self._inc = self.cast(0.5 * np.pi)

        if kwargs.get("obl", None) is not None:
            self.obl = kwargs.pop("obl")
        else:
            self._obl = self.cast(0.0)

        if kwargs.get("alpha", None) is not None:
            self.alpha = kwargs.pop("alpha")
        else:
            self._alpha = self.cast(0.0)

        self._check_kwargs("reset", kwargs)

    def X(self, **kwargs):
        """Alias for :py:meth:`design_matrix`. *Deprecated*"""
        return self.design_matrix(**kwargs)

    def design_matrix(self, **kwargs):
        """Compute and return the light curve design matrix.
        
        Args:
            xo (scalar or vector, optional): x coordinate of the occultor 
                relative to this body in units of this body's radius.
            yo (scalar or vector, optional): y coordinate of the occultor 
                relative to this body in units of this body's radius.
            zo (scalar or vector, optional): z coordinate of the occultor 
                relative to this body in units of this body's radius.
            ro (scalar, optional): Radius of the occultor in units of 
                this body's radius.
            theta (scalar or vector, optional): Angular phase of the body
                in units of :py:attr:`angle_unit`.
        """
        # Orbital kwargs
        theta, xo, yo, zo, ro = self._get_orbit(kwargs)

        # Check for invalid kwargs
        self._check_kwargs("design_matrix", kwargs)

        # Compute & return
        return self.L * self.ops.X(
            theta,
            xo,
            yo,
            zo,
            ro,
            self._inc,
            self._obl,
            self._u,
            self._f,
            self._alpha,
        )

    def intensity_design_matrix(self, lat=0, lon=0):
        """Compute and return the pixelization matrix ``P``.
        
        This matrix transforms a spherical harmonic coefficient vector
        to a vector of intensities on the surface.

        Args:
            lat (scalar or vector, optional): latitude at which to evaluate
                the design matrix in units of :py:attr:`angle_unit`.
            lon (scalar or vector, optional): longitude at which to evaluate
                the design matrix in units of :py:attr:`angle_unit`.

        .. note::
            This method ignores any filters (such as limb darkening
            or velocity weighting) and illumination (for reflected light
            maps).
        
        """
        # Get the Cartesian points
        lat, lon = vectorize(*self.cast(lat, lon))
        lat *= self._angle_factor
        lon *= self._angle_factor

        # Compute & return
        return self.L * self.ops.P(lat, lon)

    def flux(self, **kwargs):
        """
        Compute and return the light curve.

        Args:
            xo (scalar or vector, optional): x coordinate of the occultor 
                relative to this body in units of this body's radius.
            yo (scalar or vector, optional): y coordinate of the occultor 
                relative to this body in units of this body's radius.
            zo (scalar or vector, optional): z coordinate of the occultor 
                relative to this body in units of this body's radius.
            ro (scalar, optional): Radius of the occultor in units of 
                this body's radius.
            theta (scalar or vector, optional): Angular phase of the body
                in units of :py:attr:`angle_unit`.
        """
        # Orbital kwargs
        theta, xo, yo, zo, ro = self._get_orbit(kwargs)

        # Check for invalid kwargs
        self._check_kwargs("flux", kwargs)

        # Compute & return
        return self.L * self.ops.flux(
            theta,
            xo,
            yo,
            zo,
            ro,
            self._inc,
            self._obl,
            self._y,
            self._u,
            self._f,
            self._alpha,
        )

    def intensity(self, lat=0, lon=0):
        """
        Compute and return the intensity of the map.
        
        Args:
            lat (scalar or vector, optional): latitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit`.
            lon (scalar or vector, optional): longitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit``.

        """
        # Get the Cartesian points
        lat, lon = vectorize(*self.cast(lat, lon))
        lat *= self._angle_factor
        lon *= self._angle_factor

        # Compute & return
        return self.L * self.ops.intensity(lat, lon, self._y, self._u, self._f)

    def render(self, res=300, projection="ortho", theta=0.0):
        """Compute and return the intensity of the map on a grid.
        
        Returns an image of shape ``(res, res)``, unless ``theta`` is a vector,
        in which case returns an array of shape ``(nframes, res, res)``, where
        ``nframes`` is the number of values of ``theta``. However, if this is 
        a spectral map, ``nframes`` is the number of wavelength bins and 
        ``theta`` must be a scalar.
        
        Args:
            res (int, optional): The resolution of the map in pixels on a
                side. Defaults to 300.
            projection (string, optional): The map projection. Accepted
                values are ``ortho``, corresponding to an orthographic
                projection (as seen on the sky), and ``rect``, corresponding
                to an equirectangular latitude-longitude projection.
                Defaults to ``ortho``.
            theta (scalar or vector, optional): The map rotation phase in
                units of :py:attr:`angle_unit`. If this is a vector, an
                animation is generated. Defaults to ``0.0``.
        """
        # Multiple frames?
        if self.nw is not None:
            animated = True
        else:
            if is_theano(theta):
                animated = theta.ndim > 0
            else:
                animated = hasattr(theta, "__len__")

        # Convert
        projection = get_projection(projection)
        theta = vectorize(self.cast(theta) * self._angle_factor)

        # Compute
        image = self.L * self.ops.render(
            res,
            projection,
            theta,
            self._inc,
            self._obl,
            self._y,
            self._u,
            self._f,
            self._alpha,
        )

        # Squeeze?
        if animated:
            return image
        else:
            return reshape(image, [res, res])

    def show(self, **kwargs):
        """
        Display an image of the map, with optional animation. See the
        docstring of :py:meth:`render` for more details and additional
        keywords accepted by this method.

        Args:
            cmap (string or colormap instance, optional): The matplotlib colormap
                to use. Defaults to ``plasma``.
            figsize (tuple, optional): Figure size in inches. Default is 
                (3, 3) for orthographic maps and (7, 3.5) for rectangular
                maps.
            projection (string, optional): The map projection. Accepted
                values are ``ortho``, corresponding to an orthographic
                projection (as seen on the sky), and ``rect``, corresponding
                to an equirectangular latitude-longitude projection.
                Defaults to ``ortho``.
            grid (bool, optional): Show latitude/longitude grid lines?
                Defaults to True.
            interval (int, optional): Interval between frames in milliseconds
                (animated maps only). Defaults to 75.
            file (string, optional): The file name (including the extension)
                to save the animation to (animated maps only). Defaults to None.
            html5_video (bool, optional): If rendering in a Jupyter notebook,
                display as an HTML5 video? Default is True. If False, displays
                the animation using Javascript (file size will be larger.)
        """
        # Get kwargs
        cmap = kwargs.pop("cmap", "plasma")
        projection = get_projection(kwargs.get("projection", "ortho"))
        grid = kwargs.pop("grid", True)
        interval = kwargs.pop("interval", 75)
        file = kwargs.pop("file", None)
        html5_video = kwargs.pop("html5_video", True)

        # Get the map orientation
        if config.lazy:
            inc = self._inc.eval()
            obl = self._obl.eval()
        else:
            inc = self._inc
            obl = self._obl

        # Get the rotational phase
        if config.lazy:
            theta = vectorize(
                self.cast(kwargs.pop("theta", 0.0)) * self._angle_factor
            ).eval()
        else:
            theta = np.atleast_1d(
                kwargs.pop("theta", 0.0) * self._angle_factor
            )

        # Render the map if needed
        image = kwargs.pop("image", None)
        if image is None:

            # We need to evaluate the variables so we can plot the map!
            if config.lazy:

                # Get kwargs
                res = kwargs.pop("res", 300)

                # Evaluate the variables
                inc = self._inc.eval()
                obl = self._obl.eval()
                y = self._y.eval()
                u = self._u.eval()
                f = self._f.eval()
                alpha = self._alpha.eval()

                # Explicitly call the compiled version of `render`
                image = self.L.eval().reshape(-1, 1, 1) * self.ops.render(
                    res,
                    projection,
                    theta,
                    inc,
                    obl,
                    y,
                    u,
                    f,
                    alpha,
                    force_compile=True,
                )

            else:

                # Easy!
                image = self.render(theta=theta / self._angle_factor, **kwargs)
                kwargs.pop("res", None)

        if len(image.shape) == 3:
            nframes = image.shape[0]
        else:
            nframes = 1
            image = np.reshape(image, (1,) + image.shape)

        # Animation
        animated = nframes > 1

        if projection == STARRY_RECTANGULAR_PROJECTION:
            # Set up the plot
            figsize = kwargs.pop("figsize", (7, 3.75))
            fig, ax = plt.subplots(1, figsize=figsize)
            extent = (-180, 180, -90, 90)

            # Grid lines
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
            figsize = kwargs.pop("figsize", (3, 3))
            fig, ax = plt.subplots(1, figsize=figsize)
            ax.axis("off")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            extent = (-1, 1, -1, 1)

            # Grid lines
            if grid:
                x = np.linspace(-1, 1, 10000)
                y = np.sqrt(1 - x ** 2)
                ax.plot(x, y, "k-", alpha=1, lw=1)
                ax.plot(x, -y, "k-", alpha=1, lw=1)
                lat_lines = get_ortho_latitude_lines(inc=inc, obl=obl)
                for x, y in lat_lines:
                    ax.plot(x, y, "k-", lw=0.5, alpha=0.5, zorder=100)
                lon_lines = get_ortho_longitude_lines(
                    inc=inc, obl=obl, theta=theta[0]
                )
                ll = [None for n in lon_lines]
                for n, l in enumerate(lon_lines):
                    ll[n], = ax.plot(
                        l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=100
                    )

        # Plot the first frame of the image
        img = ax.imshow(
            image[0],
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="none",
            vmin=np.nanmin(image),
            vmax=np.nanmax(image),
            animated=animated,
        )

        # Display or save the image / animation
        if animated:

            def updatefig(i):
                img.set_array(image[i])
                if (
                    projection == STARRY_ORTHOGRAPHIC_PROJECTION
                    and grid
                    and len(theta) > 1
                    and self.nw is None
                ):
                    lon_lines = get_ortho_longitude_lines(
                        inc=inc, obl=obl, theta=theta[i]
                    )
                    for n, l in enumerate(lon_lines):
                        ll[n].set_xdata(l[0])
                        ll[n].set_ydata(l[1])
                    return img, ll
                else:
                    return (img,)

            ani = FuncAnimation(
                fig,
                updatefig,
                interval=interval,
                blit=False,
                frames=image.shape[0],
            )

            # Business as usual
            if (file is not None) and (file != ""):
                if file.endswith(".mp4"):
                    ani.save(file, writer="ffmpeg")
                elif file.endswith(".gif"):
                    ani.save(file, writer="imagemagick")
                else:
                    # Try and see what happens!
                    ani.save(file)
                plt.close()
            else:
                try:
                    if "zmqshell" in str(type(get_ipython())):
                        plt.close()
                        if html5_video:
                            display(HTML(ani.to_html5_video()))
                        else:
                            display(HTML(ani.to_jshtml()))
                    else:
                        raise NameError("")
                except NameError:
                    plt.show()
                    plt.close()

            # Matplotlib generates an annoying empty
            # file when producing an animation. Delete it.
            try:
                os.remove("None0000000.png")
            except FileNotFoundError:
                pass

        else:
            plt.show()

        # Check for invalid kwargs
        kwargs.pop("rv", None)
        kwargs.pop("projection", None)
        kwargs.pop("source", None)
        self._check_kwargs("show", kwargs)

    def load(
        self,
        image,
        healpix=False,
        sampling_factor=8,
        sigma=None,
        psd=True,
        **kwargs
    ):
        """Load an image, array, or ``healpix`` map. 
        
        This routine uses various routines in ``healpix`` to compute the 
        spherical harmonic expansion of the input image and sets the map's 
        :py:attr:`y` coefficients accordingly.

        Args:
            image: A path to an image file, a two-dimensional ``numpy`` 
                array, or a ``healpix`` map array (if ``healpix`` is True).
            healpix (bool, optional): Treat ``image`` as a ``healpix`` array? 
                Default is False.
            sampling_factor (int, optional): Oversampling factor when computing 
                the ``healpix`` representation of an input image or array. 
                Default is 8. Increasing this number may improve the fidelity 
                of the expanded map, but the calculation will take longer.
            sigma (float, optional): If not None, apply gaussian smoothing 
                with standard deviation ``sigma`` to smooth over 
                spurious ringing features. Smoothing is performed with 
                the ``healpix.sphtfunc.smoothalm`` method. 
                Default is None.
            psd (bool, optional): Force the map to be positive semi-definite?
                Default is True.
            kwargs (optional): Any other kwargs passed directly to
                :py:meth:`minimize` (only if ``psd`` is True).
        """
        # TODO?
        if self.nw is not None:
            raise NotImplementedError(
                "Method not available for spectral maps."
            )

        # Is this a file name?
        if type(image) is str:
            y = image2map(
                image,
                lmax=self.ydeg,
                sigma=sigma,
                sampling_factor=sampling_factor,
            )
        # or is it an array?
        elif type(image) is np.ndarray:
            if healpix:
                y = healpix2map(
                    image,
                    lmax=self.ydeg,
                    sigma=sigma,
                    sampling_factor=sampling_factor,
                )
            else:
                y = array2map(
                    image,
                    lmax=self.ydeg,
                    sigma=sigma,
                    sampling_factor=sampling_factor,
                )
        else:
            raise ValueError("Invalid `image` value.")

        # Ingest the coefficients
        self._y = self.cast(y)

        # Ensure positive semi-definite?
        if psd:

            # Find the minimum
            _, _, I = self.minimize(**kwargs)
            if config.lazy:
                I = I.eval()

            # Scale the coeffs?
            if I < 0:
                fac = 1.0 / (1.0 - np.pi * I)
                if config.lazy:
                    self._y *= fac
                    self._y = self.ops.set_map_vector(self._y, 0, 1.0)
                else:
                    self._y[1:] *= fac

    def add_spot(
        self, amp, sigma=0.1, lat=0.0, lon=0.0, preserve_luminosity=False
    ):
        r"""Add the expansion of a gaussian spot to the map.
        
        This function adds a spot whose functional form is the spherical
        harmonic expansion of a gaussian in the quantity 
        :math:`\cos\Delta\theta`, where :math:`\Delta\theta`
        is the angular separation between the center of the spot and another
        point on the surface. The spot brightness is controlled by the
        parameter ``amp``, which is defined as the fractional change in the
        total luminosity of the object due to the spot.

        Args:
            amp (scalar or vector): The amplitude of the spot. This is equal
                to the fractional change in the luminosity of the map due to
                the spot (unless ``preserve_luminosity`` is True.) If the map
                has more than one wavelength bin, this must be a vector of
                length equal to the number of wavelength bins.
            sigma (scalar, optional): The standard deviation of the gaussian. 
                Defaults to 0.1.
            lat (scalar, optional): The latitude of the spot in units of 
                :py:attr:`angle_unit`. Defaults to 0.0.
            lon (scalar, optional): The longitude of the spot in units of 
                :py:attr:`angle_unit`. Defaults to 0.0.
            preserve_luminosity (bool, optional): If True, preserves the 
                current map luminosity when adding the spot. Regions of the 
                map outside of the spot will therefore get brighter. 
                Defaults to False.
        """
        amp, _ = vectorize(self.cast(amp), np.ones(self.nw))
        sigma, lat, lon = self.cast(sigma, lat, lon)
        self._y, new_L = self.ops.add_spot(
            self._y,
            self._L,
            amp,
            sigma,
            lat * self._angle_factor,
            lon * self._angle_factor,
        )
        if not preserve_luminosity:
            self._L = new_L

    def minimize(self, **kwargs):
        r"""Find the global minimum of the map intensity.

        """
        # TODO?
        if self.nw is not None:
            raise NotImplementedError(
                "Method not available for spectral maps."
            )
        self.ops.minimize.setup()
        lat, lon, I = self.ops.get_minimum(self.y)
        return lat / self._angle_factor, lon / self._angle_factor, I


class RVBase(object):
    """The radial velocity ``starry`` map class.

    This class handles velocity-weighted intensities for use in
    Rossiter-McLaughlin effect investigations. It has all the same 
    attributes and methods as b:py:class:`starry.maps.YlmBase`, with the
    additions and modifications listed below.

    .. note::
        Instantiate this class by calling :py:func:`starry.Map` with
        ``rv`` set to True.
    """

    _ops_class_ = OpsRV

    def reset(self):
        super(RVBase, self).reset()
        self._veq = self.cast(0.0)

    @property
    def veq(self):
        """The equatorial velocity of the body in arbitrary units.
        
        .. warning::
            If this map is associated with a :py:class:`starry.Body`
            instance in a Keplerian system, changing the body's
            radius and rotation period does not currently affect this
            value. The user must explicitly change this value to affect
            the map's radial velocity.

        """
        return self._veq

    @veq.setter
    def veq(self, value):
        self._veq = value

    def _unset_RV_filter(self):
        f = np.zeros(self.Nf)
        f[0] = np.pi
        self._f = self.cast(f)

    def _set_RV_filter(self):
        self._f = self.ops.compute_rv_filter(
            self._inc, self._obl, self._veq, self._alpha
        )

    def rv(self, **kwargs):
        """Compute the net radial velocity one would measure from the object.

        The radial velocity is computed as the ratio

            :math:`\\Delta RV = \\frac{\\int Iv \\mathrm{d}A}{\\int I \\mathrm{d}A}`

        where both integrals are taken over the visible portion of the 
        projected disk. :math:`I` is the intensity field (described by the
        spherical harmonic and limb darkening coefficients) and :math:`v`
        is the radial velocity field (computed based on the equatorial velocity
        of the star, its orientation, etc.)

        Args:
            xo (scalar or vector, optional): x coordinate of the occultor 
                relative to this body in units of this body's radius.
            yo (scalar or vector, optional): y coordinate of the occultor 
                relative to this body in units of this body's radius.
            zo (scalar or vector, optional): z coordinate of the occultor 
                relative to this body in units of this body's radius.
            ro (scalar, optional): Radius of the occultor in units of 
                this body's radius.
            theta (scalar or vector, optional): Angular phase of the body
                in units of :py:attr:`angle_unit`.
        """
        # Orbital kwargs
        theta, xo, yo, zo, ro = self._get_orbit(kwargs)

        # Check for invalid kwargs
        self._check_kwargs("rv", kwargs)

        # Compute
        return self.ops.rv(
            theta,
            xo,
            yo,
            zo,
            ro,
            self._inc,
            self._obl,
            self._y,
            self._u,
            self._veq,
            self._alpha,
        )

    def intensity(self, **kwargs):
        """
        Compute and return the intensity of the map.
        
        Args:
            lat (scalar or vector, optional): latitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit`.
            lon (scalar or vector, optional): longitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit`.
            rv (bool, optional): If True, computes the velocity-weighted
                intensity instead. Defaults to True.

        """
        # Compute the velocity-weighted intensity if `rv==True`
        rv = kwargs.pop("rv", True)
        if rv:
            self._set_RV_filter()
        res = super(RVBase, self).intensity(**kwargs)
        if rv:
            self._unset_RV_filter()
        return res

    def render(self, **kwargs):
        """
        Compute and return the intensity of the map on a grid.
        
        Returns an image of shape ``(res, res)``, unless ``theta`` is a vector,
        in which case returns an array of shape ``(nframes, res, res)``, where
        ``nframes`` is the number of values of ``theta``. However, if this is 
        a spectral map, ``nframes`` is the number of wavelength bins and 
        ``theta`` must be a scalar.
        
        Args:
            res (int, optional): The resolution of the map in pixels on a
                side. Defaults to 300.
            projection (string, optional): The map projection. Accepted
                values are ``ortho``, corresponding to an orthographic
                projection (as seen on the sky), and ``rect``, corresponding
                to an equirectangular latitude-longitude projection.
                Defaults to ``ortho``.
            theta (scalar or vector, optional): The map rotation phase in
                units of :py:attr:`angle_unit`. If this is a vector, an
                animation is generated. Defaults to ``0.0``.
            rv (bool, optional): If True, computes the velocity-weighted
                intensity instead. Defaults to True.
        """
        # Render the velocity map if `rv==True`
        # Override the `projection` kwarg if we're
        # plotting the radial velocity.
        rv = kwargs.pop("rv", True)
        if rv:
            kwargs.pop("projection", None)
            self._set_RV_filter()
        res = super(RVBase, self).render(**kwargs)
        if rv:
            self._unset_RV_filter()
        return res

    def show(self, **kwargs):
        """
        Display an image of the map, with optional animation. See the
        docstring of :py:meth:`render` for more details and additional
        keywords accepted by this method.

        Args:
            cmap (string or colormap instance): The matplotlib colormap
                to use. Defaults to ``plasma``.
            projection (string, optional): The map projection. Accepted
                values are ``ortho``, corresponding to an orthographic
                projection (as seen on the sky), and ``rect``, corresponding
                to an equirectangular latitude-longitude projection.
                Defaults to ``ortho``.
            grid (bool, optional): Show latitude/longitude grid lines?
                Defaults to True.
            interval (int, optional): Interval between frames in milliseconds
                (animated maps only). Defaults to 75.
            mp4 (string, optional): The file name to save an ``mp4``
                animation to (animated maps only). Defaults to None.
            rv (bool, optional): If True, computes the velocity-weighted
                intensity instead. Defaults to True.
        """
        # Show the velocity map if `rv==True`
        # Override the `projection` kwarg if we're
        # plotting the radial velocity.
        rv = kwargs.pop("rv", True)
        if rv:
            kwargs.pop("projection", None)
            self._set_RV_filter()
        res = super(RVBase, self).show(rv=rv, **kwargs)
        if rv:
            self._unset_RV_filter()
        return res


class ReflectedBase(object):
    """The reflected light ``starry`` map class.

    This class handles light curves and phase curves of objects viewed
    in reflected light. It has all the same attributes and methods as
    :py:class:`starry.maps.YlmBase`, with the
    additions and modifications listed below.

    .. note::
        Instantiate this class by calling 
        :py:func:`starry.Map` with ``reflected`` set to True.
    """

    _ops_class_ = OpsReflected

    def design_matrix(self, **kwargs):
        """
        Compute and return the light curve design matrix.
        
        Args:
            xo (scalar or vector, optional): x coordinate of the occultor 
                relative to this body in units of this body's radius.
            yo (scalar or vector, optional): y coordinate of the occultor 
                relative to this body in units of this body's radius.
            zo (scalar or vector, optional): z coordinate of the occultor 
                relative to this body in units of this body's radius.
            ro (scalar, optional): Radius of the occultor in units of 
                this body's radius.
            theta (scalar or vector, optional): Angular phase of the body
                in units of :py:attr:`angle_unit`.
            source (vector or matrix, optional): The Cartesian position of
                the illumination source in the observer frame, where ``x`` points
                to the right on the sky, ``y`` points up on the sky, and ``z`` 
                points out of the sky toward the observer. This must be either 
                a unit vector of shape ``(3,)`` or a sequence of unit 
                vectors of shape ``(N, 3)``. Defaults to ``[-1, 0, 0]``.

        """
        # Orbital kwargs
        theta, xo, yo, zo, ro = self._get_orbit(kwargs)

        # Source position
        source = atleast_2d(self.cast(kwargs.pop("source", [-1, 0, 0])))
        theta, xo, yo, zo, source = vectorize(theta, xo, yo, zo, source)

        # Check for invalid kwargs
        self._check_kwargs("X", kwargs)

        # Compute & return
        return self.L * self.ops.X(
            theta,
            xo,
            yo,
            zo,
            ro,
            self._inc,
            self._obl,
            self._u,
            self._f,
            self._alpha,
            source,
        )

    def flux(self, **kwargs):
        """
        Compute and return the reflected flux from the map.
        
        Args:
            xo (scalar or vector, optional): x coordinate of the occultor 
                relative to this body in units of this body's radius.
            yo (scalar or vector, optional): y coordinate of the occultor 
                relative to this body in units of this body's radius.
            zo (scalar or vector, optional): z coordinate of the occultor 
                relative to this body in units of this body's radius.
            ro (scalar, optional): Radius of the occultor in units of 
                this body's radius.
            theta (scalar or vector, optional): Angular phase of the body
                in units of :py:attr:`angle_unit`.
            source (vector or matrix, optional): The Cartesian position of
                the illumination source in the observer frame, where ``x`` points
                to the right on the sky, ``y`` points up on the sky, and ``z`` 
                points out of the sky toward the observer. This must be either 
                a unit vector of shape ``(3,)`` or a sequence of unit 
                vectors of shape ``(N, 3)``. Defaults to ``[-1, 0, 0]``.
        """
        # Orbital kwargs
        theta, xo, yo, zo, ro = self._get_orbit(kwargs)

        # Source position
        source = atleast_2d(self.cast(kwargs.pop("source", [-1, 0, 0])))
        theta, xo, yo, zo, source = vectorize(theta, xo, yo, zo, source)

        # Check for invalid kwargs
        self._check_kwargs("flux", kwargs)

        # Compute & return
        return self.L * self.ops.flux(
            theta,
            xo,
            yo,
            zo,
            ro,
            self._inc,
            self._obl,
            self._y,
            self._u,
            self._f,
            self._alpha,
            source,
        )

    def intensity(self, lat=0, lon=0, source=[-1, 0, 0]):
        """
        Compute and return the intensity of the map.
        
        Args:
            lat (scalar or vector, optional): latitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit`.
            lon (scalar or vector, optional): longitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit`.
            source (vector or matrix, optional): The Cartesian position of
                the illumination source in the observer frame, where ``x`` points
                to the right on the sky, ``y`` points up on the sky, and ``z`` 
                points out of the sky toward the observer. This must be either 
                a unit vector of shape ``(3,)`` or a sequence of unit 
                vectors of shape ``(N, 3)``. Defaults to ``[-1, 0, 0]``.
        """
        # Get the Cartesian points
        lat, lon = vectorize(*self.cast(lat, lon))
        lat *= self._angle_factor
        lon *= self._angle_factor

        # Source position
        source = atleast_2d(self.cast(source))
        lat, lon, source = vectorize(lat, lon, source)

        # Compute & return
        return self.L * self.ops.intensity(
            lat, lon, self._y, self._u, self._f, source
        )

    def render(
        self, res=300, projection="ortho", theta=0.0, source=[-1, 0, 0]
    ):
        """
        Compute and return the intensity of the map on a grid.
        
        Returns an image of shape ``(res, res)``, unless ``theta`` is a vector,
        in which case returns an array of shape ``(nframes, res, res)``, where
        ``nframes`` is the number of values of ``theta``. However, if this is 
        a spectral map, ``nframes`` is the number of wavelength bins and 
        ``theta`` must be a scalar.
        
        Args:
            res (int, optional): The resolution of the map in pixels on a
                side. Defaults to 300.
            projection (string, optional): The map projection. Accepted
                values are ``ortho``, corresponding to an orthographic
                projection (as seen on the sky), and ``rect``, corresponding
                to an equirectangular latitude-longitude projection.
                Defaults to ``ortho``.
            theta (scalar or vector, optional): The map rotation phase in
                units of :py:attr:`angle_unit`. If this is a vector, an
                animation is generated. Defaults to ``0.0``.
            source (vector or matrix, optional): The Cartesian position of
                the illumination source in the observer frame, where ``x`` points
                to the right on the sky, ``y`` points up on the sky, and ``z`` 
                points out of the sky toward the observer. This must be either 
                a unit vector of shape ``(3,)`` or a sequence of unit 
                vectors of shape ``(N, 3)``. Defaults to ``[-1, 0, 0]``.
        """
        # Multiple frames?
        if self.nw is not None:
            animated = True
        else:
            if is_theano(theta):
                animated = theta.ndim > 0
            else:
                animated = hasattr(theta, "__len__")

        # Convert stuff as needed
        projection = get_projection(projection)
        theta = self.cast(theta) * self._angle_factor
        source = atleast_2d(self.cast(source))
        theta, source = vectorize(theta, source)

        # Compute
        image = self.L * self.ops.render(
            res,
            projection,
            theta,
            self._inc,
            self._obl,
            self._y,
            self._u,
            self._f,
            self._alpha,
            source,
        )

        # Squeeze?
        if animated:
            return image
        else:
            return reshape(image, [res, res])

    def show(self, **kwargs):
        # We need to evaluate the variables so we can plot the map!
        if config.lazy and kwargs.get("image", None) is None:

            # Get kwargs
            res = kwargs.get("res", 300)
            projection = get_projection(kwargs.get("projection", "ortho"))
            theta = self.cast(kwargs.pop("theta", 0.0)) * self._angle_factor
            source = atleast_2d(self.cast(kwargs.pop("source", [-1, 0, 0])))
            theta, source = vectorize(theta, source)

            # Evaluate the variables
            theta = theta.eval()
            source = source.eval()
            inc = self._inc.eval()
            obl = self._obl.eval()
            y = self._y.eval()
            u = self._u.eval()
            f = self._f.eval()
            alpha = self._alpha.eval()

            # Explicitly call the compiled version of `render`
            kwargs["image"] = self.ops.render(
                res,
                projection,
                theta,
                inc,
                obl,
                y,
                u,
                f,
                alpha,
                source,
                force_compile=True,
            )
        return super(ReflectedBase, self).show(**kwargs)


def Map(
    ydeg=0, udeg=0, drorder=0, nw=None, rv=False, reflected=False, **kwargs
):
    """A generic ``starry`` surface map.

    This function is a class factory that returns an instance of either
    :py:class:`starry.maps.YlmBase`, :py:class:`starry.maps.RVBase`, or 
    :py:class:`starry.maps.ReflectedBase`,
    depending on the arguments provided by the user. The default is
    :py:class:`starry.maps.YlmBase`. If ``rv`` is True, instantiates 
    the :py:class:`starry.maps.RVBase`
    class, and if ``reflected`` is True, instantiates the 
    :py:class:`starry.maps.ReflectedBase` class.
    
    Args:
        ydeg (int, optional): Degree of the spherical harmonic map. 
            Defaults to 0.
        udeg (int, optional): Degree of the limb darkening filter. 
            Defaults to 0.
        drorder (int, optional): Order of the differential rotation
            approximation. Defaults to 0.
        nw (int, optional): Number of wavelength bins. Defaults to None
            (for monochromatic light curves).
        rv (bool, optional): If True, enable computation of radial velocities
            for modeling the Rossiter-McLaughlin effect. Defaults to False.
        reflected (bool, optional): If True, models light curves in reflected
            light. Defaults to False.
    """

    # Check args
    ydeg = int(ydeg)
    assert ydeg >= 0, "Keyword `ydeg` must be positive."
    udeg = int(udeg)
    assert udeg >= 0, "Keyword `udeg` must be positive."
    if nw is not None:
        nw = int(nw)
        assert nw > 0, "Number of wavelength bins must be positive."

    # Default map base
    Bases = (YlmBase, MapBase)

    # Radial velocity / reflected light?
    if rv:
        Bases = (RVBase,) + Bases
        fdeg = 3
    elif reflected:
        Bases = (ReflectedBase,) + Bases
        fdeg = 1
    else:
        fdeg = 0

    # Ensure we're not doing both
    if RVBase in Bases and ReflectedBase in Bases:
        raise NotImplementedError(
            "Radial velocity maps not implemented in reflected light."
        )

    # Construct the class
    class Map(*Bases):
        def __init__(self, *args, **kwargs):
            # Once a map has been instantiated, no changes
            # to the config are allowed.
            config.freeze()
            super(Map, self).__init__(*args, **kwargs)

    return Map(ydeg, udeg, fdeg, drorder, nw, **kwargs)
