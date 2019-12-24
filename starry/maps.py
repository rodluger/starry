# -*- coding: utf-8 -*-
from . import config
from ._constants import *
from ._core import OpsYlm, OpsLD, OpsReflected, OpsRV, linalg, math
from ._indices import integers, get_ylm_inds, get_ul_inds, get_ylmw_inds
from ._plotting import (
    get_ortho_latitude_lines,
    get_ortho_longitude_lines,
    get_projection,
)
from ._sht import image2map, healpix2map, array2map
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
from astropy import units
import os
import logging

logger = logging.getLogger("starry.maps")


__all__ = [
    "Map",
    "MapBase",
    "YlmBase",
    "LimbDarkenedBase",
    "RVBase",
    "ReflectedBase",
]


class Amplitude(object):
    def __get__(self, instance, owner):
        return instance._amp

    def __set__(self, instance, value):
        instance._amp = math.cast(np.ones(instance.nw) * value)


class MapBase(object):
    """The base class for all `starry` maps."""

    # The map amplitude (just an attribute)
    amp = Amplitude()

    def _no_spectral(self):
        if self.nw is not None:  # pragma: no cover
            raise NotImplementedError(
                "Method not yet implemented for spectral maps."
            )

    def __init__(self, ydeg, udeg, fdeg, drorder, nw, **kwargs):
        # Instantiate the Theano ops class
        self.ops = self._ops_class_(ydeg, udeg, fdeg, drorder, nw, **kwargs)

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
        self._drorder = drorder

        # Basic properties
        self._inc = math.cast(0.5 * np.pi)
        self._obl = math.cast(0.0)
        self._alpha = math.cast(0.0)

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
        r"""Total number of map coefficients. *Read-only*

        This is equal to :math:`N_\mathrm{y} + N_\mathrm{u} + N_\mathrm{f}`.
        """
        return self._N

    @property
    def nw(self):
        """Number of wavelength bins. *Read-only*"""
        return self._nw

    @property
    def drorder(self):
        """Differential rotation order. *Read-only*"""
        return self._drorder

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
        if not config.quiet:
            for key in kwargs.keys():
                message = "Invalid keyword `{0}` in call to `{1}()`. Ignoring."
                message = message.format(key, method)
                logger.warning(message)

    def _get_flux_kwargs(self, kwargs):
        xo = kwargs.pop("xo", 0.0)
        yo = kwargs.pop("yo", 0.0)
        zo = kwargs.pop("zo", 1.0)
        ro = kwargs.pop("ro", 0.0)
        theta = kwargs.pop("theta", 0.0)
        theta, xo, yo, zo = math.vectorize(theta, xo, yo, zo)
        theta, xo, yo, zo, ro = math.cast(theta, xo, yo, zo, ro)
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
        self._y = math.cast(y)

        u = np.zeros(self.Nu)
        u[0] = -1.0
        self._u = math.cast(u)

        f = np.zeros(self.Nf)
        f[0] = np.pi
        self._f = math.cast(f)

        self._amp = math.cast(kwargs.pop("amp", np.ones(self.nw)))

        # Reset data and priors
        self._flux = None
        self._C = None
        self._mu = None
        self._L = None
        self._solution = None

        self._check_kwargs("reset", kwargs)

    def show(self, **kwargs):
        """
        Display an image of the map, with optional animation. See the
        docstring of :py:meth:`render` for more details and additional
        keywords accepted by this method.

        Args:
            ax (optional): A matplotlib axis instance to use. Default is
                to create a new figure.
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
            colorbar (bool, optional): Display a colorbar? Default is False.
            grid (bool, optional): Show latitude/longitude grid lines?
                Defaults to True.
            interval (int, optional): Interval between frames in milliseconds
                (animated maps only). Defaults to 75.
            file (string, optional): The file name (including the extension)
                to save the figure or animation to. Defaults to None.
            html5_video (bool, optional): If rendering in a Jupyter notebook,
                display as an HTML5 video? Default is True. If False, displays
                the animation using Javascript (file size will be larger.)
            dpi (int, optional): Image resolution in dots per square inch.
                Defaults to the value defined in ``matplotlib.rcParams``.
            bitrate (int, optional): Bitrate in kbps (animations only).
                Defaults to the value defined in ``matplotlib.rcParams``.
            norm (optional): The color normalization passed to
                ``matplotlib.pyplot.imshow``. Default is None.

        .. note::
            Pure limb-darkened maps do not accept a ``projection`` keyword.

        """
        # Get kwargs
        cmap = kwargs.pop("cmap", "plasma")
        grid = kwargs.pop("grid", True)
        interval = kwargs.pop("interval", 75)
        file = kwargs.pop("file", None)
        html5_video = kwargs.pop("html5_video", True)
        norm = kwargs.pop("norm", None)
        dpi = kwargs.pop("dpi", None)
        figsize = kwargs.pop("figsize", None)
        bitrate = kwargs.pop("bitrate", None)
        colorbar = kwargs.pop("colorbar", False)
        ax = kwargs.pop("ax", None)
        if ax is None:
            custom_ax = False
        else:
            custom_ax = True

        # Ylm-base maps only
        if not self.__props__["limbdarkened"]:

            projection = get_projection(kwargs.get("projection", "ortho"))

            # Get the map orientation
            if config.lazy:
                inc = self._inc.eval()
                obl = self._obl.eval()
            else:
                inc = self._inc
                obl = self._obl

            # Get the rotational phase
            if config.lazy:
                theta = math.vectorize(
                    math.cast(kwargs.pop("theta", 0.0)) * self._angle_factor
                ).eval()
            else:
                theta = np.atleast_1d(
                    np.array(kwargs.pop("theta", 0.0)) * self._angle_factor
                )

        else:

            inc = np.array(0.5 * np.pi)
            obl = np.array(0)
            theta = np.array([0])

        # Render the map if needed
        image = kwargs.pop("image", None)
        if image is None:

            # We need to evaluate the variables so we can plot the map!
            if config.lazy:

                # Get kwargs
                res = kwargs.pop("res", 300)

                # Evaluate the variables
                u = self._u.eval()

                if not self.__props__["limbdarkened"]:

                    inc = self._inc.eval()
                    obl = self._obl.eval()
                    y = self._y.eval()
                    f = self._f.eval()
                    alpha = self._alpha.eval()

                    # Explicitly call the compiled version of `render`
                    image = self.amp.eval().reshape(
                        -1, 1, 1
                    ) * self.ops.render(
                        res, projection, theta, inc, obl, y, u, f, alpha
                    )

                else:

                    # Explicitly call the compiled version of `render`
                    image = self.amp.eval().reshape(
                        -1, 1, 1
                    ) * self.ops.render_ld(res, u)

            else:

                # Easy!
                if not self.__props__["limbdarkened"]:
                    image = self.render(
                        theta=theta / self._angle_factor, **kwargs
                    )
                else:
                    image = self.render(**kwargs)
                kwargs.pop("res", None)

        if len(image.shape) == 3:
            nframes = image.shape[0]
        else:
            nframes = 1
            image = np.reshape(image, (1,) + image.shape)

        # Animation
        animated = nframes > 1
        borders = []
        latlines = []
        lonlines = []

        if (
            not self.__props__["limbdarkened"]
            and projection == STARRY_RECTANGULAR_PROJECTION
        ):
            # Set up the plot
            if figsize is None:
                figsize = (7, 3.75)
            if ax is None:
                fig, ax = plt.subplots(1, figsize=figsize)
            else:
                fig = ax.figure
            extent = (-180, 180, -90, 90)

            # Grid lines
            if grid:
                lats = np.linspace(-90, 90, 7)[1:-1]
                lons = np.linspace(-180, 180, 13)
                latlines = [None for n in lats]
                for n, lat in enumerate(lats):
                    latlines[n] = ax.axhline(
                        lat, color="k", lw=0.5, alpha=0.5, zorder=100
                    )
                lonlines = [None for n in lons]
                for n, lon in enumerate(lons):
                    lonlines[n] = ax.axvline(
                        lon, color="k", lw=0.5, alpha=0.5, zorder=100
                    )
            ax.set_xticks(lons)
            ax.set_yticks(lats)
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")

        else:
            # Set up the plot
            if figsize is None:
                figsize = (3, 3)
            if ax is None:
                fig, ax = plt.subplots(1, figsize=figsize)
            else:
                fig = ax.figure
            ax.axis("off")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            extent = (-1, 1, -1, 1)

            # Grid lines
            if grid:
                x = np.linspace(-1, 1, 10000)
                y = np.sqrt(1 - x ** 2)
                borders = [None, None]
                (borders[0],) = ax.plot(x, y, "k-", alpha=1, lw=1)
                (borders[1],) = ax.plot(x, -y, "k-", alpha=1, lw=1)
                lats = get_ortho_latitude_lines(inc=inc, obl=obl)
                latlines = [None for n in lats]
                for n, l in enumerate(lats):
                    (latlines[n],) = ax.plot(
                        l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=100
                    )
                lons = get_ortho_longitude_lines(
                    inc=inc, obl=obl, theta=theta[0]
                )
                lonlines = [None for n in lons]
                for n, l in enumerate(lons):
                    (lonlines[n],) = ax.plot(
                        l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=100
                    )

        # Plot the first frame of the image
        if norm is None or norm == "rv":
            vmin = np.nanmin(image)
            vmax = np.nanmax(image)
            if vmin == vmax:
                vmin -= 1e-15
                vmax += 1e-15
            if norm is None:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            elif norm == "rv":
                try:
                    norm = colors.DivergingNorm(
                        vmin=vmin, vcenter=0, vmax=vmax
                    )
                except AttributeError:  # pragma: no cover
                    # DivergingNorm was introduced in matplotlib 3.1
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
        img = ax.imshow(
            image[0],
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="none",
            animated=animated,
        )

        # Add a colorbar
        if colorbar:
            if not custom_ax:
                fig.subplots_adjust(right=0.85)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(img, cax=cax, orientation="vertical")

        # Display or save the image / animation
        if animated:

            def updatefig(i):
                img.set_array(image[i])
                if (
                    not self.__props__["limbdarkened"]
                    and projection == STARRY_ORTHOGRAPHIC_PROJECTION
                    and grid
                    and len(theta) > 1
                    and self.nw is None
                ):
                    lons = get_ortho_longitude_lines(
                        inc=inc, obl=obl, theta=theta[i]
                    )
                    for n, l in enumerate(lons):
                        lonlines[n].set_xdata(l[0])
                        lonlines[n].set_ydata(l[1])
                return (img,) + tuple(lonlines + latlines + borders)

            ani = FuncAnimation(
                fig,
                updatefig,
                interval=interval,
                blit=True,
                frames=image.shape[0],
            )

            # Business as usual
            if (file is not None) and (file != ""):
                if file.endswith(".mp4"):
                    ani.save(file, writer="ffmpeg", dpi=dpi, bitrate=bitrate)
                elif file.endswith(".gif"):
                    ani.save(
                        file, writer="imagemagick", dpi=dpi, bitrate=bitrate
                    )
                else:
                    # Try and see what happens!
                    ani.save(file, dpi=dpi, bitrate=bitrate)
                if not custom_ax:
                    plt.close()
            else:  # if not custom_ax:
                try:
                    if "zmqshell" in str(type(get_ipython())):
                        plt.close()
                        with matplotlib.rc_context(
                            {
                                "savefig.dpi": dpi
                                if dpi is not None
                                else "figure",
                                "animation.bitrate": bitrate
                                if bitrate is not None
                                else -1,
                            }
                        ):
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
            if (file is not None) and (file != ""):
                if (
                    not self.__props__["limbdarkened"]
                    and projection == STARRY_ORTHOGRAPHIC_PROJECTION
                ):
                    fig.subplots_adjust(
                        left=0.01, right=0.99, bottom=0.01, top=0.99
                    )
                fig.savefig(file)
                if not custom_ax:
                    plt.close()
            elif not custom_ax:
                plt.show()

        # Check for invalid kwargs
        if self.__props__["rv"]:
            kwargs.pop("rv", None)
        if not self.__props__["limbdarkened"]:
            kwargs.pop("projection", None)
        if self.__props__["reflected"]:
            kwargs.pop("xs", None)
            kwargs.pop("ys", None)
            kwargs.pop("zs", None)
        self._check_kwargs("show", kwargs)

    def limbdark_is_physical(self):
        """Check whether the limb darkening profile (if any) is physical.

        This method uses Sturm's theorem to ensure that the limb darkening
        intensity is positive everywhere and decreases monotonically toward
        the limb.

        Returns:
            bool: Whether or not the limb darkening profile is physical.
        """
        result = self.ops.limbdark_is_physical(self.u)
        if config.lazy:
            return result
        else:
            return bool(result)

    def set_data(self, flux, C=None, cho_C=None):
        """Set the data vector and covariance matrix.

        This method is required by the :py:meth:`solve` method, which
        analytically computes the posterior over surface maps given a
        dataset and a prior, provided both are described as multivariate
        Gaussians.

        Args:
            flux (vector): The observed light curve.
            C (scalar, vector, or matrix): The data covariance. This may be
                a scalar, in which case the noise is assumed to be
                homoscedastic, a vector, in which case the covariance
                is assumed to be diagonal, or a matrix specifying the full
                covariance of the dataset. Default is None. Either `C` or
                `cho_C` must be provided.
            cho_C (matrix): The lower Cholesky factorization of the data
                covariance matrix. Defaults to None. Either `C` or
                `cho_C` must be provided.
        """
        self._flux = math.cast(flux)
        self._C = linalg.Covariance(C, cho_C, N=self._flux.shape[0])

    def set_prior(self, *, mu=None, L=None, cho_L=None):
        """Set the prior mean and covariance of the spherical harmonic coefficients.

        This method is required by the :py:meth:`solve` method, which
        analytically computes the posterior over surface maps given a
        dataset and a prior, provided both are described as multivariate
        Gaussians.

        Note that the prior is placed on the **amplitude-weighted** coefficients,
        i.e., the quantity ``x = map.amp * map.y``. Because the first spherical
        harmonic coefficient is fixed at unity, ``x[0]`` is
        the amplitude of the map. The actual spherical harmonic coefficients
        are given by ``x / map.amp``.

        This convention allows one to linearly fit for an arbitrary map normalization
        at the same time as the spherical harmonic coefficients, while ensuring
        the ``starry`` requirement that the coefficient of the :math:`Y_{0,0}`
        harmonic is always unity.

        Args:
            mu (scalar or vector): The prior mean on the amplitude-weighted
                spherical harmonic coefficients. Default is unity for the
                first term and zero for the remaining terms. If this is a vector,
                it must have length equal to :py:attr:`Ny`.
            L (scalar, vector, or matrix): The prior covariance. This may be
                a scalar, in which case the covariance is assumed to be
                homoscedastic, a vector, in which case the covariance
                is assumed to be diagonal, or a matrix specifying the full
                prior covariance. Default is None. Either `L` or
                `cho_L` must be provided.
            cho_L (matrix): The lower Cholesky factorization of the prior
                covariance matrix. Defaults to None. Either `L` or
                `cho_L` must be provided.

        """
        if mu is None:
            mu = np.zeros(self.Ny)
            mu[0] = 1.0
            mu = math.cast(mu)
        self._mu = math.cast(mu) * math.cast(np.ones(self.Ny))
        self._L = linalg.Covariance(L, cho_L, N=self.Ny)

    def remove_prior(self):
        """Remove the prior on the map coefficients."""
        self._mu = None
        self._L = None

    def solve(self, *, design_matrix=None, **kwargs):
        """Solve the linear least-squares problem for the posterior over maps.

        This method solves the generalized least squares problem given a
        light curve and its covariance (set via the :py:meth:`set_data` method)
        and a Gaussian prior on the spherical harmonic coefficients
        (set via the :py:meth:`set_prior` method). The map amplitude and
        coefficients are set to the maximum a posteriori (MAP) solution.

        Args:
            design_matrix (matrix, optional): The flux design matrix, the
                quantity returned by :py:meth:`design_matrix`. Default is
                None, in which case this is computed based on ``kwargs``.
            kwargs (optional): Keyword arguments to be passed directly to
                :py:meth:`design_matrix`, if a design matrix is not provided.

        Returns:
            A tuple containing the posterior mean for the amplitude-weighted \
            spherical harmonic coefficients (a vector) and the Cholesky factorization \
            of the posterior covariance (a lower triangular matrix).

        .. note::
            Users may call :py:meth:`draw` to draw from the
            posterior after calling this method.
        """
        # Not implemented for spectral
        self._no_spectral()

        if self._flux is None or self._C is None:
            raise ValueError("Please provide a dataset with `set_data()`.")
        elif self._mu is None or self._L is None:
            raise ValueError("Please provide a prior with `set_prior()`.")

        # Get the design matrix & remove any amplitude weighting
        if design_matrix is None:
            design_matrix = self.design_matrix(**kwargs)
        X = math.cast(design_matrix)

        # Compute the MAP solution
        self._solution = linalg.solve(
            X, self._flux, self._C.cholesky, self._mu, self._L.inverse
        )

        # Set the amplitude and coefficients
        x, _ = self._solution
        self.amp = x[0]
        self[1:, :] = x[1:] / self.amp

        # Return the mean and covariance
        return self._solution

    def lnlike(self, *, design_matrix=None, woodbury=True, **kwargs):
        """Returns the log marginal likelihood of the data given a design matrix.

        This method computes the marginal likelihood (marginalized over the
        spherical harmonic coefficients) given a
        light curve and its covariance (set via the :py:meth:`set_data` method)
        and a Gaussian prior on the spherical harmonic coefficients
        (set via the :py:meth:`set_prior` method).

        Args:
            design_matrix (matrix, optional): The flux design matrix, the
                quantity returned by :py:meth:`design_matrix`. Default is
                None, in which case this is computed based on ``kwargs``.
            woodbury (bool, optional): Solve the linear problem using the
                Woodbury identity? Default is True. The
                `Woodbury identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_
                is used to speed up matrix operations in the case that the
                number of data points is much larger than the number of
                spherical harmonic coefficients. In this limit, it can
                speed up the code by more than an order of magnitude. Keep
                in mind that the numerical stability of the Woodbury identity
                is not great, so if you're getting strange results try
                disabling this. It's also a good idea to disable this in the
                limit of few data points and large spherical harmonic degree.
            kwargs (optional): Keyword arguments to be passed directly to
                :py:meth:`design_matrix`, if a design matrix is not provided.

        Returns:
            The log marginal likelihood, a scalar.
        """
        # Not implemented for spectral
        self._no_spectral()

        if self._flux is None or self._C is None:
            raise ValueError("Please provide a dataset with `set_data()`.")
        elif self._mu is None or self._L is None:
            raise ValueError("Please provide a prior with `set_prior()`.")

        # Get the design matrix & remove any amplitude weighting
        if design_matrix is None:
            design_matrix = self.design_matrix(**kwargs)
        X = math.cast(design_matrix)

        # Compute the likelihood
        if woodbury:
            return linalg.lnlike_woodbury(
                X,
                self._flux,
                self._C.inverse,
                self._mu,
                self._L.inverse,
                self._C.lndet,
                self._L.lndet,
            )
        else:
            return linalg.lnlike(
                X, self._flux, self._C.value, self._mu, self._L.value
            )

    @property
    def solution(self):
        r"""The posterior probability distribution for the map.

        This is a tuple containing the mean and lower Cholesky factorization of the
        covariance of the amplitude-weighted spherical harmonic coefficient vector,
        obtained by solving the regularized least-squares problem
        via the :py:meth:`solve` method.

        Note that to obtain the actual covariance matrix from the lower Cholesky
        factorization :math:`L`, simply compute :math:`L L^\top`.

        Note also that this is the posterior for the **amplitude-weighted**
        map vector. Under this convention, the map amplitude is equal to the
        first term of the vector and the spherical harmonic coefficients are
        equal to the vector normalized by the first term.
        """
        if self._solution is None:
            raise ValueError("Please call `solve()` first.")
        return self._solution

    def draw(self):
        """Draw a map from the posterior distribution.

        This method draws a random map from the posterior distribution and
        sets the :py:attr:`y` map vector and :py:attr:`amp` map amplitude
        accordingly. Users should call :py:meth:`solve` to enable this
        attribute.
        """
        if self._solution is None:
            raise ValueError("Please call `solve()` first.")

        # Fast multivariate sampling using the Cholesky factorization
        yhat, cho_ycov = self._solution
        u = math.cast(np.random.randn(self.Ny))
        x = yhat + math.dot(cho_ycov, u)
        self.amp = x[0]
        self[1:, :] = x[1:] / self.amp


class YlmBase(object):
    """The default ``starry`` map class.

    This class handles light curves and phase curves of objects in
    emitted light.

    .. note::
        Instantiate this class by calling :py:func:`starry.Map` with
        ``ydeg > 0`` and both ``rv`` and ``reflected`` set to False.
    """

    _ops_class_ = OpsYlm

    def reset(self, **kwargs):
        if kwargs.get("inc", None) is not None:
            self.inc = kwargs.pop("inc")
        else:
            self._inc = math.cast(0.5 * np.pi)

        if kwargs.get("obl", None) is not None:
            self.obl = kwargs.pop("obl")
        else:
            self._obl = math.cast(0.0)

        if kwargs.get("alpha", None) is not None:
            self.alpha = kwargs.pop("alpha")
        else:
            self._alpha = math.cast(0.0)

        super(YlmBase, self).reset(**kwargs)

    @property
    def inc(self):
        """The inclination of the rotation axis in units of :py:attr:`angle_unit`."""
        return self._inc / self._angle_factor

    @inc.setter
    def inc(self, value):
        self._inc = math.cast(value) * self._angle_factor

    @property
    def obl(self):
        """The obliquity of the rotation axis in units of :py:attr:`angle_unit`."""
        return self._obl / self._angle_factor

    @obl.setter
    def obl(self, value):
        self._obl = math.cast(value) * self._angle_factor

    @property
    def alpha(self):
        """The rotational shear coefficient, a number in the range ``[0, 1]``.

        The parameter :math:`\\alpha` is used to model linear differential
        rotation. The angular velocity at a given latitude :math:`\\theta`
        is

        :math:`\\omega = \\omega_{eq}(1 - \\alpha \\sin^2\\theta)`

        where :math:`\\omega_{eq}` is the equatorial angular velocity of
        the object.

        .. note ::

            This parameter is only used if :py:attr:`drorder` is greater
            than zero and/or radial velocity mode is enabled.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if (self._drorder == 0) and not hasattr(
            self, "rv"
        ):  # pragma: no cover
            logger.warning(
                "Parameter `drorder` is zero, so setting `alpha` has no effect."
            )
        else:
            self._alpha = math.cast(value)

    def design_matrix(self, **kwargs):
        r"""Compute and return the light curve design matrix :math:`A`.

        This matrix is used to compute the flux :math:`f` from a vector of spherical
        harmonic coefficients :math:`y` and the map amplitude :math:`a`:
        :math:`f = a A y`.

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
        theta, xo, yo, zo, ro = self._get_flux_kwargs(kwargs)

        # Check for invalid kwargs
        self._check_kwargs("design_matrix", kwargs)

        # Compute & return
        return self.ops.X(
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

        This matrix is used to compute the intensity :math:`I` from a vector of spherical
        harmonic coefficients :math:`y` and the map amplitude :math:`a`:
        :math:`I = a P y`.

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
        lat, lon = math.vectorize(*math.cast(lat, lon))
        lat *= self._angle_factor
        lon *= self._angle_factor

        # Compute & return
        return self.ops.P(lat, lon)

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
        theta, xo, yo, zo, ro = self._get_flux_kwargs(kwargs)

        # Check for invalid kwargs
        self._check_kwargs("flux", kwargs)

        # Compute & return
        return self.amp * self.ops.flux(
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

    def intensity(self, lat=0, lon=0, **kwargs):
        """
        Compute and return the intensity of the map.

        Args:
            lat (scalar or vector, optional): latitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit`.
            lon (scalar or vector, optional): longitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit``.
            theta (scalar, optional): For differentially rotating maps only,
                the angular phase at which to evaluate the intensity.
                Default 0.
            limbdarken (bool, optional): Apply limb darkening
                (only if :py:attr:`udeg` > 0)? Default True.

        """
        # Get the Cartesian points
        lat, lon = math.vectorize(*math.cast(lat, lon))
        lat *= self._angle_factor
        lon *= self._angle_factor

        # If differentially rotating, allow a `theta` keyword
        if self.drorder > 0:
            alpha_theta = math.cast(kwargs.get("theta", 0.0)) * self.alpha
            alpha_theta *= self._angle_factor
        else:
            alpha_theta = math.cast(0.0)

        # If limb-darkened, allow a `limbdarken` keyword
        if self.udeg > 0 and kwargs.pop("limbdarken", True):
            ld = np.array(True)
        else:
            ld = np.array(False)

        # Check for invalid kwargs
        self._check_kwargs("intensity", kwargs)

        # Compute & return
        return self.amp * self.ops.intensity(
            lat, lon, self._y, self._u, self._f, alpha_theta, ld
        )

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
            if config.lazy:
                animated = hasattr(theta, "ndim") and theta.ndim > 0
            else:
                animated = hasattr(theta, "__len__")

        # Convert
        projection = get_projection(projection)
        theta = math.vectorize(math.cast(theta) * self._angle_factor)

        # Compute
        if self.nw is None or config.lazy:
            amp = self.amp
        else:
            # The intensity has shape `(nw, res, res)`
            # so we must reshape `amp` to take the product correctly
            amp = self.amp[:, np.newaxis, np.newaxis]
        image = amp * self.ops.render(
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
            return math.reshape(image, [res, res])

    def load(
        self,
        image,
        healpix=False,
        nside=32,
        max_iter=3,
        sigma=None,
        force_psd=False,
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
            sigma (float, optional): If not None, apply gaussian smoothing
                with standard deviation ``sigma`` to smooth over
                spurious ringing features. Smoothing is performed with
                the ``healpix.sphtfunc.smoothalm`` method.
                Default is None.
            force_psd (bool, optional): Force the map to be positive
                semi-definite? Default is False.
            nside (int, optional): The ``NSIDE`` argument to a Healpix
                map. This controls the angular resolution of the image; increase
                this for maps with higher fidelity, at the expense of extra
                compute time. Default is 32.
            max_iter (int, optional): Maximum number of iterations in trying
                to convert the map to a Healpix array. Each iteration, the
                input array is successively doubled in size in order to
                increase the angular coverage of the input. Default is 3. If
                the number of iterations exceeds this value, an error will
                be raised.
            kwargs (optional): Any other kwargs passed directly to
                :py:meth:`minimize` (only if ``psd`` is True).
        """
        # Not implemented for spectral
        self._no_spectral()

        # Is this a file name?
        if type(image) is str:
            y = image2map(
                image,
                lmax=self.ydeg,
                sigma=sigma,
                nside=nside,
                max_iter=max_iter,
            )
        # or is it an array?
        elif type(image) is np.ndarray:
            if healpix:
                y = healpix2map(
                    image,
                    lmax=self.ydeg,
                    sigma=sigma,
                    nside=nside,
                    max_iter=max_iter,
                )
            else:
                y = array2map(
                    image,
                    lmax=self.ydeg,
                    sigma=sigma,
                    nside=nside,
                    max_iter=max_iter,
                )
        else:
            raise ValueError("Invalid `image` value.")

        # Ingest the coefficients w/ appropriate normalization
        # This ensures the map intensity will have the same normalization
        # as that of the input image
        y /= 2 * np.sqrt(np.pi)
        self._y = math.cast(y / y[0])
        self._amp = math.cast(y[0] * np.pi)

        # Ensure positive semi-definite?
        if force_psd:

            # Find the minimum
            _, _, I = self.minimize(**kwargs)
            if config.lazy:
                I = I.eval()

            # Scale the coeffs?
            if I < 0:
                fac = self._amp / (self._amp - np.pi * I)
                if config.lazy:
                    self._y *= fac
                    self._y = self.ops.set_map_vector(self._y, 0, 1.0)
                else:
                    self._y[1:] *= fac

    def rotate(self, axis, theta):
        """Rotate the current map vector an angle ``theta`` about ``axis``.

        Args:
            axis (vector): The axis about which to rotate the map.
            theta (scalar): The angle of (counter-clockwise) rotation.
        """
        axis = math.cast(axis)
        axis /= math.sqrt(math.sum(axis ** 2))
        # Note that we rotate by -theta since
        # this is the *RHS* rotation operator
        y = self.ops.dotR(
            math.transpose(
                math.reshape(self.y, (-1, 1 if self.nw is None else self.nw))
            ),
            axis[0],
            axis[1],
            axis[2],
            -math.cast(theta * self._angle_factor),
        )
        if self.nw is None:
            self._y = y[0]
        else:
            self._y = math.transpose(y)

    def add_spot(
        self,
        amp=None,
        intensity=None,
        relative=True,
        sigma=0.1,
        lat=0.0,
        lon=0.0,
    ):
        r"""Add the expansion of a gaussian spot to the map.

        This function adds a spot whose functional form is the spherical
        harmonic expansion of a gaussian in the quantity
        :math:`\cos\Delta\theta`, where :math:`\Delta\theta`
        is the angular separation between the center of the spot and another
        point on the surface. The spot brightness is controlled by either the
        parameter ``amp``, defined as the fractional change in the
        total luminosity of the object due to the spot, or the parameter
        ``intensity``, defined as the fractional change in the
        intensity at the center of the spot.

        Args:
            amp (scalar or vector, optional): The amplitude of the spot. This
                is equal to the fractional change in the luminosity of the map
                due to the spot. If the map has more than one wavelength bin,
                this must be a vector of length equal to the number of
                wavelength bins. Default is None.
                Either ``amp`` or ``intensity`` must be given.
            intensity (scalar or vector, optional): The intensity of the spot.
                This is equal to the fractional change in the intensity of the
                map at the *center* of the spot. If the map has more than one
                wavelength bin, this must be a vector of length equal to the
                number of wavelength bins. Default is None.
                Either ``amp`` or ``intensity`` must be given.
            relative (bool, optional): If True, computes the spot expansion
                assuming the fractional `amp` or `intensity` change is relative
                to the **current** map amplitude/intensity. If False, computes
                the spot expansion assuming the fractional change is relative
                to the **original** map amplitude/intensity (i.e., that of
                a featureless map). Defaults to True. Note that if True,
                adding two spots with the same values of `amp` or `intensity`
                will generally result in *different* intensities at their
                centers, since the first spot will have changed the map
                intensity everywhere! Defaults to True.
            sigma (scalar, optional): The standard deviation of the gaussian.
                Defaults to 0.1.
            lat (scalar, optional): The latitude of the spot in units of
                :py:attr:`angle_unit`. Defaults to 0.0.
            lon (scalar, optional): The longitude of the spot in units of
                :py:attr:`angle_unit`. Defaults to 0.0.

        """
        # Parse the amplitude
        if (amp is None and intensity is None) or (
            amp is not None and intensity is not None
        ):
            raise ValueError("Please provide either `amp` or `intensity`.")
        elif amp is not None:
            amp, _ = math.vectorize(math.cast(amp), np.ones(self.nw))
            # Normalize?
            if not relative:
                amp /= self.amp
        else:
            # Vectorize if needed
            intensity, _ = math.vectorize(
                math.cast(intensity), np.ones(self.nw)
            )
            # Normalize?
            if not relative:
                baseline = 1.0 / np.pi
            else:
                baseline = self.intensity(lat=lat, lon=lon, limbdarken=False)
            DeltaI = baseline * intensity
            # The integral of the gaussian in cos(Delta theta) over the
            # surface of the sphere is sigma * sqrt(2 * pi^3). Combining
            # this with the normalization convention of starry (a factor of 4),
            # the corresponding spot amplitude is...
            amp = sigma * np.sqrt(2 * np.pi ** 3) * DeltaI / 4
            if not relative:
                amp /= self.amp

        # Parse remaining kwargs
        sigma, lat, lon = math.cast(sigma, lat, lon)

        # Get the Ylm expansion of the spot. Note that yspot[0] is not
        # unity, so we'll need to normalize it before setting self._y
        yspot = self.ops.expand_spot(
            amp, sigma, lat * self._angle_factor, lon * self._angle_factor
        )
        y_new = self._y + yspot
        amp_new = self._amp * y_new[0]
        y_new /= y_new[0]

        # Update the map and the normalizing amplitude
        self._y = y_new
        self._amp = amp_new

    def minimize(self, oversample=1, ntries=1, return_info=False):
        """Find the global minimum of the map intensity.

        Args:
            oversample (int): Factor by which to oversample the initial
                grid on which the brute force search is performed. Default 1.
            ntries (int): Number of times the nonlinear minimizer is called.
                Default 1.
            return_info (bool): Return the info from the minimization call?
                Default is False.

        Returns:
            A tuple of the latitude, longitude, and the value of the intensity \
            at the minimum. If ``return_info`` is True, also returns the detailed \
            solver information.
        """
        # Not implemented for spectral
        self._no_spectral()

        self.ops._minimize.setup(oversample=oversample, ntries=ntries)
        lat, lon, I = self.ops.get_minimum(self.y)
        if return_info:  # pragma: no cover
            return (
                lat / self._angle_factor,
                lon / self._angle_factor,
                self._amp * I,
                self.ops._minimize.result,
            )
        else:
            return (
                lat / self._angle_factor,
                lon / self._angle_factor,
                self._amp * I,
            )


class LimbDarkenedBase(object):
    """The ``starry`` map class for purely limb-darkened maps.

    This class handles light curves of purely limb-darkened objects in
    emitted light.

    .. note::
        Instantiate this class by calling :py:func:`starry.Map` with
        ``ydeg`` set to zero and both ``rv`` and ``reflected`` set to False.
    """

    _ops_class_ = OpsLD

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
        """
        # Orbital kwargs
        theta = kwargs.pop("theta", None)
        _, xo, yo, zo, ro = self._get_flux_kwargs(kwargs)

        # Check for invalid kwargs
        if theta is not None:
            # If the user passed in `theta`, make sure a warning is raised
            kwargs["theta"] = theta
        self._check_kwargs("flux", kwargs)

        # Compute & return
        return self.amp * self.ops.flux(xo, yo, zo, ro, self._u)

    def intensity(self, mu=None, x=None, y=None):
        r"""
        Compute and return the intensity of the map.

        Args:
            mu (scalar or vector, optional): the radial parameter :math:`\mu`,
                equal to the cosine of the angle between the line of sight and
                the normal to the surface. Default is None.
            x (scalar or vector, optional): the Cartesian x position on the
                surface in units of the body's radius. Default is None.
            y (scalar or vector, optional): the Cartesian y position on the
                surface in units of the body's radius. Default is None.

        .. note::
            Users must provide either `mu` **or** `x` and `y`.
        """
        # Get the Cartesian points
        if mu is not None:
            mu = math.vectorize(math.cast(mu))
            assert (
                x is None and y is None
            ), "Please provide either `mu` or `x` and `y`, but not both."
        else:
            assert (
                x is not None and y is not None
            ), "Please provide either `mu` or `x` and `y`."
            x, y = math.vectorize(*math.cast(x, y))
            mu = (1 - x ** 2 - y ** 2) ** 0.5

        # Compute & return
        return self.amp * self.ops.intensity(mu, self._u)

    def render(self, res=300):
        """Compute and return the intensity of the map on a grid.

        Returns an image of shape ``(res, res)``.

        Args:
            res (int, optional): The resolution of the map in pixels on a
                side. Defaults to 300.
        """
        # Multiple frames?
        if self.nw is not None:
            animated = True
        else:
            animated = False

        # Compute
        image = self.amp * self.ops.render_ld(res, self._u)

        # Squeeze?
        if animated:
            return image
        else:
            return math.reshape(image, [res, res])


class RVBase(object):
    """The radial velocity ``starry`` map class.

    This class handles velocity-weighted intensities for use in
    Rossiter-McLaughlin effect investigations. It has all the same
    attributes and methods as :py:class:`starry.maps.YlmBase`, with the
    additions and modifications listed below.

    .. note::
        Instantiate this class by calling :py:func:`starry.Map` with
        ``ydeg > 0`` and ``rv`` set to True.
    """

    _ops_class_ = OpsRV

    def reset(self, **kwargs):
        self.velocity_unit = kwargs.pop("velocity_unit", units.m / units.s)
        self.veq = kwargs.pop("veq", 0.0)
        super(RVBase, self).reset(**kwargs)

    @property
    def velocity_unit(self):
        """An ``astropy.units`` unit defining the velocity metric for this map."""
        return self._velocity_unit

    @velocity_unit.setter
    def velocity_unit(self, value):
        assert value.physical_type == "speed"
        self._velocity_unit = value
        self._velocity_factor = value.in_units(units.m / units.s)

    @property
    def veq(self):
        """The equatorial velocity of the body in units of :py:attr:`velocity_unit`.

        .. warning::
            If this map is associated with a :py:class:`starry.Body`
            instance in a Keplerian system, changing the body's
            radius and rotation period does not currently affect this
            value. The user must explicitly change this value to affect
            the map's radial velocity.

        """
        return self._veq / self._velocity_factor

    @veq.setter
    def veq(self, value):
        self._veq = math.cast(value) * self._velocity_factor

    def _unset_RV_filter(self):
        f = np.zeros(self.Nf)
        f[0] = np.pi
        self._f = math.cast(f)

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
        theta, xo, yo, zo, ro = self._get_flux_kwargs(kwargs)

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
            theta (scalar, optional): For differentially rotating maps only,
                the angular phase at which to evaluate the intensity.
                Default 0.
            limbdarken (bool, optional): Apply limb darkening
                (only if :py:attr:`udeg` > 0)? Default True.

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

    def show(self, rv=True, **kwargs):
        # Show the velocity map if `rv==True`
        # Override some kwargs if we're
        # plotting the radial velocity.
        if rv:
            kwargs.pop("projection", None)
            self._set_RV_filter()
            kwargs["cmap"] = kwargs.pop("cmap", "RdBu_r")
            kwargs["norm"] = kwargs.pop("norm", "rv")
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

    The spherical harmonic coefficients of a map in reflected light are
    an expansion of the object's *albedo* (instead of its emissivity, in
    the default case).

    The illumination source is currently assumed to be a point source for
    the purposes of computing the illumination profile on the surface of the
    body. However, if the illumination source occults the body, the flux
    *is* computed correctly (i.e., the occulting body has a finite radius).
    This approximation holds if the distance between the occultor and the source
    is large compared to the size of the source. It fails, for example, in the
    case of an extremely short-period planet, in which case signficantly more
    than half the planet surface is illuminated by the star at any given time.
    We plan to account for this effect in the future, so stay tuned.

    The ``xo``, ``yo``, and ``zo`` parameters in several of the methods below
    specify the position of the illumination source in units of this body's
    radius. The flux returned by the :py:meth:`flux` method is normalized such
    that when the distance between the occultor and the illumination source is
    unity, a uniform unit-amplitude map will emit a flux of unity when viewed
    at noon.

    This class does not currently support occultations. If an occultation
    does occur, a ``ValueError`` will be raised. Support for occultations in
    reflected light will be added in an upcoming version, so stay tuned.

    .. note::
        Instantiate this class by calling
        :py:func:`starry.Map` with ``ydeg > 0`` and ``reflected`` set to True.
    """

    _ops_class_ = OpsReflected

    def _get_flux_kwargs(self, kwargs):
        xo = kwargs.pop("xo", 0.0)
        yo = kwargs.pop("yo", 0.0)
        zo = kwargs.pop("zo", 1.0)
        ro = kwargs.pop("ro", 0.0)
        xs = kwargs.pop("xs", 0.0)
        ys = kwargs.pop("ys", 0.0)
        zs = kwargs.pop("zs", 1.0)
        theta = kwargs.pop("theta", 0.0)
        theta, xs, ys, zs, xo, yo, zo = math.vectorize(
            theta, xs, ys, zs, xo, yo, zo
        )
        theta, xs, ys, zs, xo, yo, zo, ro = math.cast(
            theta, xs, ys, zs, xo, yo, zo, ro
        )
        theta *= self._angle_factor
        return theta, xs, ys, zs, xo, yo, zo, ro

    def design_matrix(self, **kwargs):
        r"""
        Compute and return the light curve design matrix, :math:`A`.

        This matrix is used to compute the flux :math:`f` from a vector of spherical
        harmonic coefficients :math:`y` and the map amplitude :math:`a`:
        :math:`f = a A y`.

        Args:
            xs (scalar or vector, optional): x coordinate of the illumination
                source relative to this body in units of this body's radius.
            ys (scalar or vector, optional): y coordinate of the illumination
                source relative to this body in units of this body's radius.
            zs (scalar or vector, optional): z coordinate of the illumination
                source relative to this body in units of this body's radius.
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

        .. note::
            ``starry`` does not yet support occultations in reflected light.

        """
        # Orbital kwargs
        theta, xs, ys, zs, xo, yo, zo, ro = self._get_flux_kwargs(kwargs)

        # Check for invalid kwargs
        self._check_kwargs("X", kwargs)

        # Compute & return
        return self.ops.X(
            theta,
            xs,
            ys,
            zs,
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

    def flux(self, **kwargs):
        """
        Compute and return the reflected flux from the map.

        Args:
            xs (scalar or vector, optional): x coordinate of the illumination
                source relative to this body in units of this body's radius.
            ys (scalar or vector, optional): y coordinate of the illumination
                source relative to this body in units of this body's radius.
            zs (scalar or vector, optional): z coordinate of the illumination
                source relative to this body in units of this body's radius.
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

        .. note::
            ``starry`` does not yet support occultations in reflected light.

        """
        # Orbital kwargs
        theta, xs, ys, zs, xo, yo, zo, ro = self._get_flux_kwargs(kwargs)

        # Check for invalid kwargs
        self._check_kwargs("flux", kwargs)

        # Compute & return
        return self.amp * self.ops.flux(
            theta,
            xs,
            ys,
            zs,
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

    def intensity(self, lat=0, lon=0, xs=0, ys=0, zs=1, **kwargs):
        """
        Compute and return the intensity of the map.

        Args:
            lat (scalar or vector, optional): latitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit`.
            lon (scalar or vector, optional): longitude at which to evaluate
                the intensity in units of :py:attr:`angle_unit`.
            xs (scalar or vector, optional): x coordinate of the illumination
                source relative to this body in units of this body's radius.
            ys (scalar or vector, optional): y coordinate of the illumination
                source relative to this body in units of this body's radius.
            zs (scalar or vector, optional): z coordinate of the illumination
                source relative to this body in units of this body's radius.
            theta (scalar, optional): For differentially rotating maps only,
                the angular phase at which to evaluate the intensity.
                Default 0.
            limbdarken (bool, optional): Apply limb darkening
                (only if :py:attr:`udeg` > 0)? Default True.
        """
        # Get the Cartesian points
        lat, lon = math.vectorize(*math.cast(lat, lon))
        lat *= self._angle_factor
        lon *= self._angle_factor

        # Get the source position
        xs, ys, zs = math.vectorize(*math.cast(xs, ys, zs))

        # Get the amplitude
        if self.nw is None or config.lazy:
            amp = self.amp
        else:
            # The intensity has shape `(nsurf_pts, nw, nsource_pts)`
            # so we must reshape `amp` to take the product correctly
            amp = self.amp[np.newaxis, :, np.newaxis]

        # If differentially rotating, allow a `theta` keyword
        if self.drorder > 0:
            alpha_theta = math.cast(kwargs.get("theta", 0.0)) * self.alpha
            alpha_theta *= self._angle_factor
        else:
            alpha_theta = math.cast(0.0)
            self._check_kwargs("intensity", kwargs)

        # If limb-darkened, allow a `limbdarken` keyword
        if self.udeg > 0:
            ld = np.array(True)
        else:
            ld = np.array(False)

        # Compute & return
        return amp * self.ops.intensity(
            lat, lon, self._y, self._u, self._f, xs, ys, zs, alpha_theta, ld
        )

    def render(
        self,
        res=300,
        projection="ortho",
        illuminate=True,
        theta=0.0,
        xs=0,
        ys=0,
        zs=1,
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
            illuminate (bool, optional): Illuminate the map? Default is True.
            theta (scalar or vector, optional): The map rotation phase in
                units of :py:attr:`angle_unit`. If this is a vector, an
                animation is generated. Defaults to ``0.0``.
            xs (scalar or vector, optional): x coordinate of the illumination
                source relative to this body in units of this body's radius.
            ys (scalar or vector, optional): y coordinate of the illumination
                source relative to this body in units of this body's radius.
            zs (scalar or vector, optional): z coordinate of the illumination
                source relative to this body in units of this body's radius.

        """
        # Multiple frames?
        if self.nw is not None:
            animated = True
        else:
            if config.lazy:
                animated = hasattr(theta, "ndim") and theta.ndim > 0
            else:
                animated = hasattr(theta, "__len__")

        # Convert stuff as needed
        projection = get_projection(projection)
        theta = math.cast(theta) * self._angle_factor
        xs = math.cast(xs)
        ys = math.cast(ys)
        zs = math.cast(zs)
        theta, xs, ys, zs = math.vectorize(theta, xs, ys, zs)
        illuminate = int(illuminate)

        # Compute
        if self.nw is None or config.lazy:
            amp = self.amp
        else:
            # The intensity has shape `(nw, res, res)`
            # so we must reshape `amp` to take the product correctly
            amp = self.amp[:, np.newaxis, np.newaxis]

        image = amp * self.ops.render(
            res,
            projection,
            illuminate,
            theta,
            self._inc,
            self._obl,
            self._y,
            self._u,
            self._f,
            self._alpha,
            xs,
            ys,
            zs,
        )

        # Squeeze?
        if animated:
            return image
        else:
            return math.reshape(image, [res, res])

    def show(self, **kwargs):
        # We need to evaluate the variables so we can plot the map!
        if config.lazy and kwargs.get("image", None) is None:

            # Get kwargs
            res = kwargs.pop("res", 300)
            projection = get_projection(kwargs.get("projection", "ortho"))
            theta = math.cast(kwargs.pop("theta", 0.0)) * self._angle_factor
            xs = math.cast(kwargs.pop("xs", 0))
            ys = math.cast(kwargs.pop("ys", 0))
            zs = math.cast(kwargs.pop("zs", 1))
            theta, xs, ys, zs = math.vectorize(theta, xs, ys, zs)
            illuminate = int(kwargs.pop("illuminate", True))

            # Evaluate the variables
            theta = theta.eval()
            xs = xs.eval()
            ys = ys.eval()
            zs = zs.eval()
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
                illuminate,
                theta,
                inc,
                obl,
                y,
                u,
                f,
                alpha,
                xs,
                ys,
                zs,
            )
            kwargs["theta"] = theta / self._angle_factor

        return super(ReflectedBase, self).show(**kwargs)


def Map(
    ydeg=0, udeg=0, drorder=0, nw=None, rv=False, reflected=False, **kwargs
):
    """A generic ``starry`` surface map.

    This function is a class factory that returns either
    a :doc:`spherical harmonic map <SphericalHarmonicMap>`,
    a :doc:`limb darkened map <LimbDarkenedMap>`,
    a :doc:`radial velocity map <RadialVelocityMap>`, or
    a :doc:`reflected light map <ReflectedLightMap>`,
    depending on the arguments provided by the user. The default is
    a :doc:`spherical harmonic map <SphericalHarmonicMap>`. If ``rv`` is True,
    instantiates a :doc:`radial velocity map <RadialVelocityMap>` map, and
    if ``reflected`` is True, instantiates a :doc:`reflected light map
    <ReflectedLightMap>`. Otherwise, if ``ydeg`` is zero, instantiates a
    :doc:`limb darkened map <LimbDarkenedMap>`.

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
    drorder = int(drorder)
    assert (drorder >= 0) and (
        drorder <= 3
    ), "Differential rotation orders above 3 are not supported."
    if drorder > 0:
        assert ydeg > 0, "Differential rotation requires `ydeg` >= 1."

        # TODO: phase this next warning out
        logger.warning(
            "Differential rotation is still an experimental feature. "
            "Use it with care."
        )

        Ddeg = (4 * drorder + 1) * ydeg
        if Ddeg >= 50:
            logger.warning(
                "The degree of the differential rotation operator "
                "is currently {0}, ".format(Ddeg)
                + "which will likely cause the code to run very slowly. "
                "Consider decreasing the degree of the map or the order "
                "of differential rotation."
            )

    # Limb-darkened?
    if (ydeg == 0) and (rv is False) and (reflected is False):

        # TODO: Add support for wavelength-dependent limb darkening
        if nw is not None:
            raise NotImplementedError(
                "Multi-wavelength limb-darkened maps are not yet supported."
            )

        Bases = (LimbDarkenedBase, MapBase)
    else:
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
    if rv and reflected:
        raise NotImplementedError(
            "Radial velocity maps not implemented in reflected light."
        )

    # Construct the class
    class Map(*Bases):

        # Tags
        __props__ = dict(
            limbdarkened=LimbDarkenedBase in Bases,
            reflected=ReflectedBase in Bases,
            rv=RVBase in Bases,
            spectral=nw is not None,
            differential_rotation=drorder > 0,
        )

        def __init__(self, *args, **kwargs):
            # Once a map has been instantiated, no changes
            # to the config are allowed.
            config.freeze()
            super(Map, self).__init__(*args, **kwargs)

    return Map(ydeg, udeg, fdeg, drorder, nw, **kwargs)
