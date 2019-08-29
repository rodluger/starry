# -*- coding: utf-8 -*-
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
)
from .indices import get_ylm_inds, get_ul_inds, get_ylmw_inds
from .utils import get_ortho_latitude_lines, get_ortho_longitude_lines
from .sht import image2map, healpix2map, array2map
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from warnings import warn
from astropy import units


__all__ = ["Map", "MapBase", "YlmBase", "RVBase", "ReflectedBase"]


class Luminosity(object):
    """Descriptor for map luminosity."""

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

    def __init__(self, ydeg, udeg, fdeg, nw, quiet=False):
        """

        """
        # Instantiate the Theano ops class
        self.quiet = quiet
        self.ops = self._ops_class_(ydeg, udeg, fdeg, nw, quiet=quiet)
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

        # Luminosity
        self._L = self.cast(np.ones(self.nw))

        # Units
        self.angle_unit = units.degree

        # Initialize
        self.reset()

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
        """Number of spherical harmonic coefficients. *Read-only*

        This is equal to :math:`(y_\mathrm{deg} + 1)^2`.
        """
        return self._Ny

    @property
    def udeg(self):
        """Limb darkening degree. *Read-only*"""
        return self._udeg

    @property
    def Nu(self):
        """Number of limb darkening coefficients, including :math:`u_0`. *Read-only*
        
        This is equal to :math:`u_\mathrm{deg} + 1`.
        """
        return self._Nu

    @property
    def fdeg(self):
        """Degree of the multiplicative filter. *Read-only*"""
        return self._fdeg

    @property
    def Nf(self):
        """Number of spherical harmonic coefficients in the filter. *Read-only*

        This is equal to :math:`(f_\mathrm{deg} + 1)^2`.
        """
        return self._Nf

    @property
    def deg(self):
        """Total degree of the map. *Read-only*
        
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
    def axis(self):
        """A unit vector representing the axis of rotation for the map."""
        return self.ops.get_axis(self._inc, self._obl)

    @axis.setter
    def axis(self, axis):
        axis = self.cast(axis)
        inc_obl = self.ops.get_inc_obl(axis)
        self._inc = inc_obl[0]
        self._obl = inc_obl[1]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int, slice)):
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
        if isinstance(idx, (int, np.int, slice)):
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

    def reset(self):
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

        self._inc = self.cast(np.pi / 2)
        self._obl = self.cast(0.0)

        self._prot = 1.0
        self._t0 = self.cast(0.0)
        self._r = 1.0
        self._veq = 0.0

    def X(self, **kwargs):
        """Alias for :py:meth:`design_matrix`. *Deprecated*"""
        return self.design_matrix(**kwargs)

    def design_matrix(self, **kwargs):
        """Compute and return the light curve design matrix.
        
        Args:
            xo (array or scalar, optional): x coordinate of the occultor 
                relative to this body in units of this body's radius.
            yo (array or scalar, optional): y coordinate of the occultor 
                relative to this body in units of this body's radius.
            zo (array or scalar, optional): z coordinate of the occultor 
                relative to this body in units of this body's radius.
            ro (scalar, optional): Radius of the occultor in units of 
                this body's radius.
            theta (array or scalar, optional): Angular phase of the body
                in units of :py:attr:`angle_unit`.
        """
        # Orbital kwargs
        theta, xo, yo, zo, ro = self._get_orbit(kwargs)

        # Check for invalid kwargs
        self._check_kwargs("design_matrix", kwargs)

        # Compute & return
        return self.L * self.ops.X(
            theta, xo, yo, zo, ro, self._inc, self._obl, self._u, self._f
        )

    def intensity_design_matrix(self, **kwargs):
        """Compute and return the pixelization matrix ``P``.
        
        .. note::
            This method ignores any filters (such as limb darkening
            or velocity weighting) and illumination (for reflected light
            maps).
        
        """
        # Get the Cartesian points
        lat = kwargs.pop("lat", None)
        lon = kwargs.pop("lon", None)
        if lat is None and lon is None:
            x = kwargs.pop("x", 0.0)
            y = kwargs.pop("y", 0.0)
            z = kwargs.pop("z", None)
            if z is not None:
                x, y, z = vectorize(*self.cast(x, y, z))
            else:
                x, y = vectorize(*self.cast(x, y))
                z = (1.0 - x ** 2 - y ** 2) ** 0.5
        else:
            lat, lon = vectorize(*self.cast(lat, lon))
            lat *= self._angle_factor
            lon *= self._angle_factor
            xyz = self.ops.latlon_to_xyz(self.axis, lat, lon)
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]

        # Check for invalid kwargs
        self._check_kwargs("intensity_design_matrix", kwargs)

        # Compute & return
        return self.L * self.ops.P(x, y, z)

    def flux(self, **kwargs):
        """
        Compute and return the light curve.

        Args:
            xo (array or scalar, optional): x coordinate of the occultor 
                relative to this body in units of this body's radius.
            yo (array or scalar, optional): y coordinate of the occultor 
                relative to this body in units of this body's radius.
            zo (array or scalar, optional): z coordinate of the occultor 
                relative to this body in units of this body's radius.
            ro (scalar, optional): Radius of the occultor in units of 
                this body's radius.
            theta (array or scalar, optional): Angular phase of the body
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
        )

    def intensity(self, **kwargs):
        """
        Compute and return the intensity of the map
        at a given ``(lat, lon)`` or ``(x, y, z)``
        point on the surface.
        
        """
        # Get the Cartesian points
        lat = kwargs.pop("lat", None)
        lon = kwargs.pop("lon", None)
        if lat is None and lon is None:
            x = kwargs.pop("x", 0.0)
            y = kwargs.pop("y", 0.0)
            z = kwargs.pop("z", None)
            if z is not None:
                x, y, z = vectorize(*self.cast(x, y, z))
            else:
                x, y = vectorize(*self.cast(x, y))
                z = (1.0 - x ** 2 - y ** 2) ** 0.5
        else:
            lat, lon = vectorize(*self.cast(lat, lon))
            lat *= self._angle_factor
            lon *= self._angle_factor
            xyz = self.ops.latlon_to_xyz(self.axis, lat, lon)
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]

        # Check for invalid kwargs
        self._check_kwargs("intensity", kwargs)

        # Compute & return
        return self.L * self.ops.intensity(x, y, z, self._y, self._u, self._f)

    def render(self, res=300, projection="ortho", theta=0.0):
        """
        Compute and return the sky-projected intensity of 
        the map on a square Cartesian grid.
        
        The shape of the returned image is `(nframes, res, res)`.

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
        )

        # Squeeze?
        if animated:
            return image
        else:
            return reshape(image, [res, res])

    def show(self, **kwargs):
        """
        Display an image of the map, with optional animation.

        """
        # Get kwargs
        cmap = kwargs.pop("cmap", "plasma")
        projection = get_projection(kwargs.get("projection", "ortho"))
        grid = kwargs.pop("grid", True)
        interval = kwargs.pop("interval", 75)
        mp4 = kwargs.pop("mp4", None)

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
            fig, ax = plt.subplots(1, figsize=(7, 3.75))
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
            fig, ax = plt.subplots(1, figsize=(3, 3))
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
                if grid and len(theta) > 1:
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
            if (mp4 is not None) and (mp4 != ""):
                if mp4.endswith(".mp4"):
                    mp4 = mp4[:-4]
                ani.save("%s.mp4" % mp4, writer="ffmpeg")
                plt.close()
            else:
                try:
                    if "zmqshell" in str(type(get_ipython())):
                        plt.close()
                        display(HTML(ani.to_jshtml()))
                    else:
                        raise NameError("")
                except NameError:
                    plt.show()
                    plt.close()
        else:
            plt.show()

        # Check for invalid kwargs
        kwargs.pop("rv", None)
        kwargs.pop("projection", None)
        self._check_kwargs("show", kwargs)

    def load(self, image, healpix=False, sampling_factor=8, sigma=None):
        """Load an image, array, or ``healpix`` map. 
        
        This routine uses various routines in ``healpix`` to compute the 
        spherical harmonic expansion of the input image and sets the map's 
        :py:attr:`y` coefficients accordingly.

        The map is oriented such that the north pole of the input image
        is placed at the north pole of the current rotation axis.

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
        """
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

        # Align the map with the axis of rotation
        self.align([0, 1, 0])

    def rotate(self, theta, axis=None):
        """

        """
        # Get inc and obl
        if axis is None:
            axis = self.ops.get_axis(self._inc, self._obl)
        else:
            axis = self.cast(axis)

        # Cast to tensor & convert to internal units
        theta = self.cast(theta) * self._angle_factor

        # Rotate
        self._y = self.ops.rotate(axis, theta, self._y)

    def align(self, source=None, dest=None):
        """Rotate the map to align ``source`` with ``dest``.

        The standard way of rotating maps in ``starry`` is to
        provide the axis and angle of rotation, but this isn't always
        convenient. In some cases, it is easier to specify a source
        point/axis and a destination point/axis, and rotate the map such 
        that the source aligns with the destination. This is particularly 
        useful for changing map projections. For instance, to view the 
        map pole-on,

        .. code-block:: python

            map.align(source=map.axis, dest=[0, 0, 1])

        This rotates the map axis to align with the z-axis, which points
        toward the observer.

        Another useful application is if you want the map to rotate along with
        the axis of rotation when you change the map's inclination and/or
        obliquity. In other words, say you specified the coefficients of the 
        map in the default frame (in which the rotation axis points along 
        :math:`\\hat{y}`) but the object you're modeling is inclined/rotated 
        with respect to the plane of the sky. After specifying the map's 
        inclination and obliquity, run

        .. code-block:: python

            map.align(source=[0, 1, 0], dest=map.axis)
        
        to align the pole of the map with the axis of rotation. You are now
        in the frame corresponding to the plane of the sky.

        Args:
            source (array, optional): A unit vector describing the source 
                position. This point will be rotated onto ``dest``. Default 
                is the current map axis.
            dest (array, optional): A unit vector describing the destination 
                position. The ``source`` point will be rotated onto this point. 
                Default is the current map axis.
        """
        if source is None:
            source = self.axis
        if dest is None:
            dest = self.axis
        source = self.cast(source)
        dest = self.cast(dest)
        self._y = self.ops.align(self._y, source, dest)

    def add_spot(self, amp, sigma=0.1, lat=0.0, lon=0.0):
        """
        
        """
        amp, _ = vectorize(self.cast(amp), np.ones(self.nw))
        sigma, lat, lon = self.cast(sigma, lat, lon)
        self._y = self.ops.add_spot(
            self._y,
            amp,
            sigma,
            lat * self._angle_factor,
            lon * self._angle_factor,
            self._inc,
            self._obl,
        )


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
        self._alpha = self.cast(0.0)
        self._veq = self.cast(0.0)

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

    @property
    def veq(self):
        """The equatorial velocity of the body in arbitrary units."""
        # TODO: Warn user that changing `prot` and `r` doesn't currently
        # affect this value.
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
        """
        Compute the net radial velocity one would measure from the object.

        The radial velocity is computed as the ratio

            :math:`\\Delta RV = \\frac{\\int Iv \\mathrm{d}A}{\\int I \\mathrm{d}A}`

        where both integrals are taken over the visible portion of the 
        projected disk. :math:`I` is the intensity field (described by the
        spherical harmonic and limb darkening coefficients) and :math:`v`
        is the radial velocity field (computed based on the equatorial velocity
        of the star, its orientation, etc.)
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
            source,
        )

    def flux(self, **kwargs):
        """
        
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
            source,
        )

    def intensity(self, **kwargs):
        """
        
        """
        # Get the Cartesian points
        lat = kwargs.pop("lat", None)
        lon = kwargs.pop("lon", None)
        if lat is None and lon is None:
            x = kwargs.pop("x", 0.0)
            y = kwargs.pop("y", 0.0)
            z = kwargs.pop("z", 1.0)
            x, y, z = vectorize(*self.cast(x, y, z))
        else:
            lat, lon = vectorize(*self.cast(lat, lon))
            lat *= self._angle_factor
            lon *= self._angle_factor
            xyz = self.ops.latlon_to_xyz(self.axis, lat, lon)
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]

        # Source position
        source = atleast_2d(self.cast(kwargs.pop("source", [-1, 0, 0])))
        x, y, z, source = vectorize(x, y, z, source)

        # Check for invalid kwargs
        self._check_kwargs("intensity", kwargs)

        # Compute & return
        return self.L * self.ops.intensity(
            x, y, z, self._y, self._u, self._f, source
        )

    def render(
        self, res=300, projection="ortho", theta=0.0, source=[-1, 0, 0]
    ):
        """
        
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
                source,
                force_compile=True,
            )
        return super(ReflectedBase, self).show(**kwargs)


def Map(ydeg=0, udeg=0, nw=None, rv=False, reflected=False, quiet=False):
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
        nw (int, optional): Number of wavelength bins. Defaults to None
            (for monochromatic light curves).
        rv (bool, optional): If True, enable computation of radial velocities
            for modeling the Rossiter-McLaughlin effect. Defaults to False.
        reflected (bool, optional): If True, models light curves in reflected
            light. Defaults to False.
        quiet (bool, optional): Suppress all logging messages? 
            Defaults to False.
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
    kwargs = dict(quiet=quiet)

    # Radial velocity / reflected light?
    if rv:
        Bases = (RVBase,) + Bases
        fdeg = 3
    elif reflected:
        Bases = (ReflectedBase,) + Bases
        fdeg = 0  # DEBUG
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

    return Map(ydeg, udeg, fdeg, nw, **kwargs)
