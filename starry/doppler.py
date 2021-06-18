# -*- coding: utf-8 -*-
from . import config
from ._constants import *
from ._core import OpsDoppler, math
from ._core.utils import is_tensor, CompileLogMessage
from ._indices import integers, get_ylm_inds, get_ylmw_inds, get_ul_inds
from .compat import evaluator
from .maps import YlmBase, MapBase, Map
from .doppler_visualize import Visualize
import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import csr_matrix
from warnings import warn
import os
import matplotlib.pyplot as plt


class DopplerMap:
    """
    Class representing a spectral-spatial stellar surface.

    Instances of this class can be used to model the spatial and wavelength
    dependence of stellar surfaces and the spectral timeseries one would
    observe in the presence of rotation-induced Doppler shifts. This class
    is specifically suited to Doppler imaging studies. It is similar (but
    different in important ways; see below) to the traditional
    ``starry.Map`` class.

    Important notes:

        - The spherical harmonic coefficients cannot be set directly.
          All spatial and spectral information must be provided via the
          :py:meth:`load` method.
        - Currently, this class does not model occultations.

    Args:
        ydeg (int, optional): Degree of the spherical harmonic map.
            Default is 1.
        udeg (int, optional): Degree of the limb darkening filter.
            Default is 0.
        nc (int, optional): Number of spectral components. The surface is
            represented as the outer product of :py:attr:`nc` spherical
            harmonic coefficient vectors and :py:attr:`nc` associated
            spectra. Default is 1.
        nt (int, optional): Number of epochs. This is the number of spectra
            one wishes to model. Default is 10.
        wav (ndarray, optional): Wavelength grid on which the data is defined
            (and on which the model will be computed) in nanometers. Default
            is a grid of ``200`` bins uniformly spaced between ``642.5 nm`` and
            ``643.5 nm``, the region spanning the classical FeI 6430 line used
            in Doppler imaging.
        wav0 (ndarray, optional): Wavelength grid on which the rest frame
            spectrum (the "template") is defined. This can be the same as
            :py:attr:`wav`, but it is strongly recommended to pad it on
            either side to avoid edge effects. If :py:attr:`wav0` is
            insufficiently padded, the values of the model near the edges
            of the wavelength grid will be incorrect, since they include
            contribution from Doppler-shifted spectral features beyond the
            edge of the grid. By default, if :py:attr:`wav0` is not provided
            or set to :py:obj:`None`, ``starry`` will compute the required
            amount of padding automatically. See :py:attr:`vsini_max` below
            for more info.
        oversample (float, optional): The oversampling factor for the internal
            wavelength grid. The number of spectral bins used to perform the
            convolutions is equal to this value times the length of
            :py:attr:`wav`. Default is 1.
        interpolate (bool, optional): The wavelength grid used internally is
            different from :py:attr:`wav`, since ``starry`` requires a grid
            that is uniformly spaced in the log of the wavelength. By default,
            :py:attr:`interpolate` is True, in which case ``starry``
            interpolates to the user-defined wavelength grids (:py:attr:`wav`
            and :py:attr:`wav0`) when any method or property is accessed.
            This incurs a slight slowdown in the computation, so users can
            obtain higher efficiency by setting this to False and handling the
            interpolation on their own. If False, all spectra returned by the
            methods and properties of this class will instead be defined on the
            :py:attr:`wav_` and :py:attr:`wav0_` grids.
        interp_order (int, optional): Order of the spline interpolation between
            the internal (:py:attr:`wav_` and :py:attr:`wav0_`) and
            user-facing (:py:attr:`wav` and :py:attr:`wav0`) wavelength
            grids. Default is 1 (linear).
        interp_tol (float, optional): For splines with
            :py:attr:`interp_order` greater than one, the interpolation
            can be made much faster by zeroing out elements of
            the interpolation operator whose absolute value is smaller than
            some small tolerance given by :py:attr:`interp_tol`.
            Default is ``1e-12``.
        vsini_max (float, optional): Maximum value of ``vsini`` for this map.
            This sets the size of the convolution kernel (and the amount of
            padding required in :py:attr:`wav0`) and should be adjusted
            to ensure that ``map.veq * sin(map.inc)`` is never larger than
            this quantity. Lower values of this quantity will result in faster
            evaluation times. Default is ``100 km/s``.
        angle_unit (``astropy.units.Unit``, optional): The unit used for
            angular quantities. Default ``deg``.
        velocity_unit (``astropy.units.Unit``, optional): The unit used for
            velocity quantities. Default ``m/s``.
        inc (scalar, optional): Inclination of the star in units of
            :py:attr:`angle_unit`. Default is ``90.0``.
        obl (scalar, optional): Obliquity of the star in units of
            :py:attr:`angle_unit`. Default is ``0.0``.
        veq (scalar, optional): Equatorial rotational velocity of the star
            in units of :py:attr:`velocity_unit`. Default is ``0.0``.


    """

    # Constants
    _clight = 299792458.0  # m/s
    _default_vsini_max = 1e5  # m/s
    _default_wav = np.linspace(642.5, 643.5, 200)  # FeI 6430

    def _default_spectrum(self):
        """
        The default spectrum to use if the user doesn't provide one.

        This consists of a single narrow Gaussian absorption line at the central
        wavelength of the first spectral component, and unity for all
        other components.

        At 643nm, this corresponds to a line with FWHM ~ 10 km/s.
        """
        return self._math.concatenate(
            (
                self._math.reshape(
                    1
                    - 0.5
                    * self._math.exp(
                        -0.5 * (self._wav0_int - self._wavr) ** 2 / 0.02 ** 2
                    ),
                    (1, self._nw0_int),
                ),
                self._math.ones((self._nc - 1, self._nw0_int)),
            ),
            axis=0,
        )

    def __init__(
        self,
        ydeg=1,
        udeg=0,
        nc=1,
        nt=10,
        wav=None,
        wav0=None,
        oversample=1,
        interpolate=True,
        interp_order=1,
        interp_tol=1e-12,
        vsini_max=None,
        lazy=None,
        **kwargs,
    ):
        # Check args
        ydeg = int(ydeg)
        assert ydeg >= 1, "Keyword ``ydeg`` must be >= 1."
        udeg = int(udeg)
        assert udeg >= 0, "Keyword ``udeg`` must be positive."
        assert nc is not None, "Please specify the number of map components."
        nc = int(nc)
        assert nc > 0, "Number of map components must be positive."
        nt = int(nt)
        assert nt > 0, "Number of epochs must be positive."
        assert (
            interp_order >= 1 and interp_order <= 5
        ), "Keyword ``interp_order`` must be in the range [1, 5]."
        if lazy is None:
            lazy = config.lazy
        self._lazy = lazy
        if lazy:
            self._math = math.lazy_math
            self._linalg = math.lazy_linalg
        else:
            self._math = math.greedy_math
            self._linalg = math.greedy_linalg

        # Dimensions
        self._ydeg = ydeg
        self._Ny = (ydeg + 1) ** 2
        self._udeg = udeg
        self._Nu = udeg + 1
        self._N = (ydeg + udeg + 1) ** 2
        self._nc = nc
        self._nt = nt

        # This parameter determines the convolution kernel width
        self.velocity_unit = kwargs.pop("velocity_unit", units.m / units.s)
        if vsini_max is None:
            vsini_max = self._default_vsini_max
        else:
            vsini_max *= self._velocity_factor
        self.vsini_max = self._math.cast(vsini_max)

        # Compute the user-facing data wavelength grid (wav)
        assert not is_tensor(
            wav
        ), "Wavelength grids must be numerical quantities."
        if wav is None:
            if wav0 is None:
                wav = self._default_wav
            else:
                raise ValueError(
                    "Please provide the observed spectrum wavelength grid `wav`."
                )
        wav = np.array(wav)
        self._wav = self._math.cast(wav)
        self._nw = len(wav)

        # Compute the size of the internal wavelength grid
        # Note that this must be odd!
        nw = int(len(wav) * oversample)
        if (nw % 2) == 0:
            nw += 1

        # Compute the internal wavelength grid (wav_int)
        wav1 = np.min(wav)
        wav2 = np.max(wav)
        wavr = np.exp(0.5 * (np.log(wav1) + np.log(wav2)))
        log_wav = np.linspace(np.log(wav1 / wavr), np.log(wav2 / wavr), nw)
        wav_int = wavr * np.exp(log_wav)
        self._wavr = wavr
        self._wav_int = self._math.cast(wav_int)
        self._nw_int = nw
        self._oversample = oversample

        # Compute the padded internal wavelength grid (wav0_int).
        # We add bins corresponding to the maximum kernel width to each
        # end of wav_int to prevent edge effects
        dlam = log_wav[1] - log_wav[0]
        betasini_max = vsini_max / self._clight
        # TODO: There used to be a 0.5 multiplying `hw` below, but that
        # seemed to cause the padding to be almost exactly half of what
        # it should be. Verify that the current expression is correct.
        hw = np.array(
            np.ceil(
                np.abs(np.log((1 + betasini_max) / (1 - betasini_max))) / dlam
            ),
            dtype="int32",
        )
        x = np.arange(0, hw + 1) * dlam
        pad_l = log_wav[0] - hw * dlam + x[:-1]
        pad_r = log_wav[-1] + x[1:]
        log_wav0_int = np.concatenate([pad_l, log_wav, pad_r])
        wav0_int = wavr * np.exp(log_wav0_int)
        nwp = len(log_wav0_int)
        self._log_wav0_int = self._math.cast(log_wav0_int)
        self._wav0_int = self._math.cast(wav0_int)
        self._nw0_int = nwp

        # Compute the user-facing rest spectrum wavelength grid (wav0)
        assert not is_tensor(
            wav0
        ), "Wavelength grids must be numerical quantities."
        if wav0 is None:
            # The default grid is the data wavelength grid with
            # a bit of padding on either side
            delta_wav = np.median(np.diff(np.sort(wav)))
            pad_l = np.arange(wav1, wav0_int[0] - delta_wav, -delta_wav)
            pad_l = pad_l[::-1][:-1]
            pad_r = np.arange(wav2, wav0_int[-1] + delta_wav, delta_wav)
            pad_r = pad_r[1:]
            wav0 = np.concatenate([pad_l, wav, pad_r])
        wav0 = np.array(wav0)
        nw0 = len(wav0)
        self._wav0 = self._math.cast(wav0)
        self._nw0 = nw0
        if (wav0_int[0] < np.min(wav0)) or (wav0_int[-1] > np.max(wav0)):
            warn(
                "Rest frame wavelength grid ``wav0`` is not sufficiently padded. "
                "Edge effects may occur. See the documentation for mode details."
            )

        # Interpolation between internal grid and user grid

        # TODO: Integrate over each wavelength bin in the output!
        # Currently we're assuming the spectra are measured in delta
        # function bins, but it should really be an integral. This
        # should be easy to bake into the interpolation matrix:
        # compute the matrix for a high resolution output grid and
        # just dot a trapezoidal integration matrix into it. The
        # resulting matrix will have the same shape, so there's no
        # loss in efficiency.

        self._interp = interpolate
        self._interp_order = interp_order
        self._interp_tol = interp_tol
        if self._interp:

            # Compute the flux interpolation operator (wav <-- wav_int)
            # ``S`` interpolates the flux back onto the user-facing ``wav`` grid
            # ``SBlock`` interpolates the design matrix onto the ``wav grid``
            S = self._get_spline_operator(wav_int, wav)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._S = self._math.sparse_cast(S.T)
            self._SBlock = self._math.sparse_cast(
                sparse_block_diag([S for n in range(nt)], format="csr")
            )
            self._STrBlock = self._math.sparse_cast(
                sparse_block_diag([S.T for n in range(nt)], format="csr")
            )

            # Compute the spec interpolation operator (wav0 <-- wav0_int)
            # ``S0`` interpolates the user-provided spectrum onto the internal grid
            # ``S0Inv`` performs the inverse operation
            S = self._get_spline_operator(wav0_int, wav0)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._S0 = self._math.sparse_cast(S.T)
            S = self._get_spline_operator(wav0, wav0_int)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._S0Inv = self._math.sparse_cast(S.T)

        else:

            # No interpolation. User-facing grids *are* the internal grids
            # User will handle interpolation on their own
            self._wav = self._wav_int
            self._wav0 = self._wav0_int
            self._nw0 = self._nw0_int
            self._nw = self._nw_int

        # Instantiate the Theano ops classs
        self.ops = OpsDoppler(
            ydeg,
            udeg,
            nw,
            nc,
            nt,
            hw,
            vsini_max,
            self._clight,
            log_wav0_int,
            **kwargs,
        )

        # Support map (for certain operations like ``load``, ``show``, etc.)
        # This map reflects all the properties of the DopplerMap except
        # the spherical harmonic coefficients ``y``; these are set on an
        # as-needed basis.
        _quiet = config.quiet
        config.quiet = True
        self._map = Map(ydeg=self.ydeg, udeg=self.udeg, lazy=self.lazy)
        config.quiet = _quiet

        # Initialize
        self.reset(**kwargs)

    @property
    def lazy(self):
        """Map evaluation mode -- lazy or greedy?"""
        return self._lazy

    @property
    def nc(self):
        """Number of spectro-spatial map components. *Read-only*"""
        return self._nc

    @property
    def nt(self):
        """Number of spectral epochs. *Read-only*"""
        return self._nt

    @property
    def nw(self):
        """Length of the user-facing flux wavelength grid :py:attr:`wav`.
        *Read-only*

        """
        return self._nw

    @property
    def nw0(self):
        """Length of the user-facing rest frame spectrum wavelength grid
        :py:attr:`wav0`. *Read-only*

        """
        return self._nw0

    @property
    def nw_(self):
        """Length of the *internal* flux wavelength grid :py:attr:`wav_`.
        *Read-only*

        """
        return self._nw_int

    @property
    def nw0_(self):
        """Length of the *internal* rest frame spectrum wavelength grid
        :py:attr:`wav0_`. *Read-only*

        """
        return self._nw0_int

    @property
    def oversample(self):
        """Spectrum oversampling factor. *Read-only*"""
        return self._oversample

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
        """Degree of the limb darkening applied to the star. *Read-only*"""
        return self._udeg

    @property
    def Nu(self):
        r"""Number of limb darkening coefficients, including :math:`u_0`.
        *Read-only*

        This is equal to :math:`u_\mathrm{deg} + 1`.
        """
        return self._Nu

    @property
    def deg(self):
        r"""Total degree of the map. *Read-only*

        This is equal to :math:`y_\mathrm{deg} + u_\mathrm{deg}`.
        """
        return self._deg

    @property
    def N(self):
        r"""Total number of map coefficients. *Read-only*

        This is equal to :math:`N_\mathrm{y} + N_\mathrm{u}`.
        """
        return self._N

    @property
    def angle_unit(self):
        """An ``astropy.units`` unit defining the angle metric for this map."""
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, value):
        assert value.physical_type == "angle"
        self._angle_unit = value
        self._angle_factor = value.in_units(units.radian)
        self._map.angle_unit = value

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
    def inc(self):
        """The inclination of the rotation axis in units of
        :py:attr:`angle_unit`.

        """
        return self._inc / self._angle_factor

    @inc.setter
    def inc(self, value):
        self._inc = self._math.cast(value) * self._angle_factor
        self._map._inc = self._inc

    @property
    def obl(self):
        """The obliquity of the rotation axis in units of
        :py:attr:`angle_unit`.

        """
        return self._obl / self._angle_factor

    @obl.setter
    def obl(self, value):
        self._obl = self._math.cast(value) * self._angle_factor

    @property
    def veq(self):
        """The equatorial velocity of the body in units of
        :py:attr:`velocity_unit`.

        """
        return self._veq / self._velocity_factor

    @veq.setter
    def veq(self, value):
        self._veq = self._math.cast(value) * self._velocity_factor

    @property
    def vsini(self):
        """
        The projected equatorial radial velocity in units of
        :py:attr:`velocity_unit`. *Read-only*

        """
        return self._veq * self._math.sin(self._inc) / self._velocity_factor

    @property
    def wav(self):
        """
        The wavelength grid for the spectral timeseries model. *Read-only*

        This is the wavelength grid on which quantities like the
        :py:meth:`flux` and :py:meth:`design_matrix` are defined.

        """
        return self._wav

    @property
    def wav_(self):
        """
        The *internal* model wavelength grid. *Read-only*

        """
        return self._wav_int

    @property
    def wav0(self):
        """
        The rest-frame wavelength grid. *Read-only*

        This is the wavelength grid on which the :py:attr:`spectrum`
        is defined.

        """
        return self._wav0

    @property
    def wav0_(self):
        """
        The *internal* rest frame spectrum wavelength grid. *Read-only*

        This is the wavelength grid on which the :py:attr:`spectrum_`
        is defined.

        """
        return self._wav0_int

    @property
    def spectrum(self):
        """
        The rest frame spectrum for each component. *Read only*

        This quantity is defined on the wavelength grid :py:attr:`wav0`.

        """
        # Interpolate to the ``wav0`` grid
        if self._interp:
            return self._math.sparse_dot(self._spectrum, self._S0)
        else:
            return self._spectrum

    @property
    def spectrum_(self):
        """
        The *internal* rest frame spectrum for each component. *Read only*

        This is defined on the wavelength grid :py:attr:`wav0_`.

        """
        return self._spectrum

    @property
    def continuum(self):
        """
        The continuum level for each spectral component. *Read only*

        """
        return self._continuum

    @property
    def y(self):
        """The spherical harmonic coefficient matrix. *Read-only*

        """
        return self._math.squeeze(self._y)

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
    def spectral_map(self):
        """
        The spectral-spatial map vector. *Read only*

        This is equal to the (unrolled) outer product of the spherical harmonic
        decompositions and their corresponding spectral components.
        Dot the design matrix into this quantity to obtain the observed
        spectral timeseries (the :py:meth:`flux`).
        """
        # Outer product with the map
        return self._math.reshape(
            self._math.dot(self._y, self._spectrum), (-1,)
        )

    def __getitem__(self, idx):
        """
        Return the spherical harmonic or limb darkening coefficient(s).

        """
        if isinstance(idx, integers) or isinstance(idx, slice):
            # User is accessing a limb darkening index
            inds = get_ul_inds(self.udeg, idx)
            return self._u[inds]
        elif isinstance(idx, tuple) and len(idx) == 2 and self.nc == 1:
            # User is accessing a Ylm index
            inds = get_ylm_inds(self.ydeg, idx[0], idx[1])
            return self._y[inds, 0]
        elif isinstance(idx, tuple) and len(idx) == 3:
            # User is accessing a Ylmc index
            inds = get_ylmw_inds(self.ydeg, self.nc, idx[0], idx[1], idx[2])
            if self.nc == 1:
                assert np.array_equal(
                    inds[1].reshape(-1), [0]
                ), "Invalid map component."
            return self._y[inds]
        else:
            raise ValueError("Invalid map index.")

    def __setitem__(self, idx, val):
        """
        Set the limb darkening coefficient(s).

        Note that this class does not support setting spherical harmonic
        coefficients directly. Please use the :py:attr:`load` method instead.

        """
        if not is_tensor(val):
            val = np.array(val)
        if isinstance(idx, integers) or isinstance(idx, slice):
            # User is accessing a limb darkening index
            inds = get_ul_inds(self.udeg, idx)
            if 0 in inds:
                raise ValueError("The u_0 coefficient cannot be set.")
            self._u = self.ops.set_vector(self._u, inds, val)
            self._map._u = self._u
        else:
            raise ValueError("Invalid map index.")

    def reset(self, **kwargs):
        """Reset all map coefficients and attributes.

        """
        # Units
        self.angle_unit = kwargs.pop("angle_unit", units.degree)
        self.velocity_unit = kwargs.pop("velocity_unit", units.m / units.s)

        # Map properties
        y = np.zeros((self._Ny, self._nc))
        y[0, :] = 1.0
        self._y = self._math.cast(y)
        u = np.zeros(self._Nu)
        u[0] = -1.0
        self._u = self._math.cast(u)
        self._map._u = self._u

        # Reset the spectrum
        self._spectrum = self._default_spectrum()
        self._continuum = self._math.cast(np.ones(self.nc))

        # Basic properties
        self.inc = kwargs.pop("inc", 0.5 * np.pi / self._angle_factor)
        self.obl = kwargs.pop("obl", 0.0)
        self.veq = kwargs.pop("veq", 0.0)

    def _get_SHT_matrix(self, nlat, nlon, eps=1e-12, smoothing=None):
        """
        Return the SHT matrix for transforming a lat-lon intensity
        grid to a vector of spherical harmonic coefficients.

        """
        # Get the lat-lon grid
        lon = np.linspace(-180, 180, nlon) * np.pi / 180
        lat = np.linspace(-90, 90, nlat) * np.pi / 180
        lon, lat = np.meshgrid(lon, lat)
        lon = lon.flatten()
        lat = lat.flatten()

        # Compute the cos(lat)-weighted SHT
        w = np.cos(lat)
        P = self.ops.P(lat, lon)
        PTSinv = P.T * (w ** 2)[None, :]
        Q = np.linalg.solve(PTSinv @ P + eps * np.eye(P.shape[1]), PTSinv)
        if smoothing is None:
            smoothing = 2.0 / self.ydeg
        if smoothing > 0:
            l = np.concatenate(
                [np.repeat(l, 2 * l + 1) for l in range(self.ydeg + 1)]
            )
            s = np.exp(-0.5 * l * (l + 1) * smoothing ** 2)
            Q *= s[:, None]
        return Q

    def load(
        self,
        *,
        images=None,
        spectra=None,
        cube=None,
        continuum="max",
        smoothing=None,
        fac=1.0,
        eps=1e-12,
    ):
        """Load a spatial and/or spectral representation of a stellar surface.

        The data cube should have shape (``nlat``, ``nlon``, :py:attr:`nw0`),
        where ``nlat`` is the number of pixels in latitude, ``nlon`` is the
        number of pixels in longitude, and :py:attr:`nw0` is the number of
        wavelength bins in the rest frame spectrum.

        This routine performs a simple spherical harmonic transform (SHT)
        to compute the spherical harmonic expansion

        Args:
            images (str or ndarray, optional):
            spectra (ndarray, optional):
            cube (ndarray, optional):
            continuum (ndarray, optional):
            smoothing (float, optional): Gaussian smoothing strength.
                Increase this value to suppress ringing or explicitly set to zero to
                disable smoothing. Default is ``2/self.ydeg``.
            fac (float, optional): Factor by which to oversample the image
                when applying the SHT. Default is ``1.0``. Increase this
                number for higher fidelity (at the expense of increased
                computational time).
            eps (float, optional): Regularization strength for the spherical
                harmonic transform. Default is ``1e-12``.
        """
        if images is not None or spectra is not None:

            # Input checks
            assert (
                cube is None
            ), "Cannot specify (`images` or `spectra`) and `cube` simultaneously."

            if images is not None:

                # ------------------------------
                # User is loading spatial images
                # ------------------------------

                # Input checks
                if type(images) is str:
                    assert self.nc == 1, "Must provide one map per component."
                    images = [images]
                elif type(images) in (tuple, list):
                    images = list(images)
                    assert (
                        len(images) == self.nc
                    ), "Must provide one map per component."
                elif type(images) is np.ndarray:
                    if self.nc == 1:
                        if images.ndim == 2:
                            images = np.array([images])
                    assert (
                        images.shape[0] == self.nc
                    ), "Must provide one map per component."
                else:
                    raise TypeError("Invalid type for `images`.")

                # Process each map
                Q = np.empty((self.Ny, 0))
                y = np.zeros((self.Ny, self.nc))
                y[:, 0] = 1.0
                for n, image in enumerate(images):

                    # Is this a file name or an array?
                    if type(image) is str:

                        # Get the full path
                        if not os.path.exists(image):
                            image = os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                "img",
                                image,
                            )
                            if not image.endswith(".png"):
                                image += ".png"
                            if not os.path.exists(image):
                                raise ValueError("File not found: %s." % image)

                        # Load the image into an ndarray
                        image = plt.imread(image)

                        # If it's an integer, normalize to [0-1]
                        # (if it's a float, it's already normalized)
                        if np.issubdtype(image.dtype, np.integer):
                            image = image / 255.0

                        # Convert to grayscale
                        if len(image.shape) == 3:
                            image = np.mean(image[:, :, :3], axis=2)
                        elif len(image.shape) == 4:
                            # ignore any transparency
                            image = np.mean(image[:, :, :3], axis=(2, 3))

                        # Re-orient
                        image = np.flipud(image)

                    elif type(image) is np.ndarray:

                        assert (
                            image.ndim == 2
                        ), "Each map must be a 2-dimensional array."

                    elif image is None:

                        continue

                    else:

                        raise TypeError(
                            "Invalid type for one of the `images`."
                        )

                    # Compute the SHT matrix if needed
                    nlat, nlon = image.shape
                    if Q.shape[1] != nlat * nlon:
                        Q = self._get_SHT_matrix(
                            nlat, nlon, eps=eps, smoothing=smoothing
                        )

                    # The Ylm coefficients are just a linear op on the image
                    # Note that we need to apply the starry 1/pi normalization
                    y[:, n] = Q @ image.reshape(nlat * nlon) / np.pi

                # Ingest the coeffs
                self._y = self._math.cast(y)

            if spectra is not None:

                # -----------------------------------
                # User is loading spectral components
                # -----------------------------------

                # Cast & reshape
                if self._nc == 1:
                    spectra = self._math.reshape(
                        self._math.cast(spectra), (1, self._nw0)
                    )
                else:
                    spectra = self.ops.enforce_shape(
                        self._math.cast(spectra),
                        np.array([self._nc, self._nw0]),
                    )

                # Interpolate from the ``wav0`` grid to internal, padded grid
                if self._interp:
                    self._spectrum = self._math.sparse_dot(
                        spectra, self._S0Inv
                    )
                else:
                    self._spectrum = VT

                # Compute the continuum
                if type(continuum) is str:
                    if continuum.startswith("max"):
                        continuum = np.max(spectra, axis=1)
                    elif continuum.startswith("min"):
                        continuum = np.min(spectra, axis=1)
                    elif continuum.startswith("med"):
                        continuum = np.median(spectra, axis=1)
                    elif continuum.startswith("mode"):
                        values, counts = np.unique(
                            spectra, axis=1, return_counts=True
                        )
                        continuum = values[:, np.argmax(counts)]
                    else:
                        raise ValueError(
                            "Invalid value for argument `continuum`."
                        )
                else:
                    assert not is_tensor(
                        continuum
                    ), "Argument `continuum` cannot be a tensor."
                    continuum = np.array(continuum)
                    assert (
                        continuum.ndim == 1 and len(continuum) == self.nc
                    ), "Argument `continuum` must be a vector of length `nc`."
                self._continuum = self._math.cast(continuum)

        elif cube is not None:

            # --------------------------------
            # User is loading a full data cube
            # --------------------------------

            # Input checks
            assert not is_tensor(cube)
            assert cube.ndim == 3
            assert cube.shape[2] == self.nw0
            nlat = cube.shape[0]
            nlon = cube.shape[1]
            if type(continuum) is str:
                if continuum.startswith("max"):
                    continuum = np.max(cube, axis=2)
                elif continuum.startswith("min"):
                    continuum = np.min(cube, axis=2)
                elif continuum.startswith("med"):
                    continuum = np.median(cube, axis=2)
                elif continuum.startswith("mode"):
                    # Warning: quite slow!
                    values, counts = np.unique(
                        cube, axis=2, return_counts=True
                    )
                    continuum = values[:, :, np.argmax(counts)]
                else:
                    raise ValueError("Invalid value for argument `continuum`.")
            else:
                assert not is_tensor(
                    continuum
                ), "Argument `continuum` cannot be a tensor."
                assert continuum.shape == (
                    nlat,
                    nlon,
                ), "Argument `continuum` must have shape `(cube.shape[0], cube.shape[1])."

            # Singular value decomposition
            M = cube.reshape(nlat * nlon, self.nw0)
            U, s, VT = np.linalg.svd(M, full_matrices=False)

            # Keep only `nc` components; absorb singular values into `U`
            U = U[:, : self.nc]
            VT = VT[: self.nc, :]
            s = s[: self.nc]
            U = U * s

            # --------------------------------------
            # The spectra are just the rows of `V^T`
            # --------------------------------------

            # Interpolate from the ``wav0`` grid to internal, padded grid
            if self._interp:
                self._spectrum = self._math.sparse_dot(VT, self._S0Inv)
            else:
                self._spectrum = VT

            # -----------------------------------------------
            # The spatial maps are just the components of `U`
            # -----------------------------------------------

            # Downsample the image to Nyquist-ish
            factor = fac * 4 * self.ydeg / nlat
            if factor < 1:
                U = zoom(
                    U.reshape(nlat, nlon, self.nc),
                    [factor, factor, 1],
                    mode="nearest",
                )
                continuum = zoom(continuum, [factor, factor], mode="nearest")
                nlat, nlon = U.shape[0], U.shape[1]
                U = U.reshape(nlat * nlon, self.nc)

            # Get the lat-lon grid
            Q = self._get_SHT_matrix(nlat, nlon, eps=eps, smoothing=smoothing)

            # The Ylm coefficients are just a linear op on the image
            # Note that we need to apply the starry 1/pi normalization
            y = Q @ U / np.pi
            self._y = self._math.reshape(
                self._math.cast(y), (self.Ny, self.nc)
            )

            # Compute the continuum level for each component
            self._continuum = self._math.cast(
                np.linalg.lstsq(U, continuum.reshape(-1))[0]
            )

    def _get_spline_operator(self, input_grid, output_grid):
        """

        """
        assert not is_tensor(
            input_grid, output_grid
        ), "Wavelength grids must be numerical quantities."
        S = np.zeros((len(output_grid), len(input_grid)))
        for n in range(len(input_grid)):
            y = np.zeros_like(input_grid)
            y[n] = 1.0
            S[:, n] = Spline(input_grid, y, k=self._interp_order)(output_grid)
        return S

    def _get_default_theta(self, theta):
        """

        """
        if theta is None:
            theta = self._math.cast(
                np.linspace(0, 2 * np.pi, self._nt, endpoint=False)
            )
        else:
            theta = (
                self.ops.enforce_shape(
                    self._math.cast(theta), np.array([self._nt])
                )
                * self._angle_factor
            )
        return theta

    def baseline_matrix(self, theta=None):
        theta = self._get_default_theta(theta) / self._angle_factor
        B = self._map.design_matrix(theta=theta)
        B = self._math.tile(B, [1, self.nc])
        B *= np.repeat(self._continuum, self.Ny)
        B = self._math.repeat(B, self.nw, axis=0)
        return B

    def design_matrix(self, theta=None, fix_spectrum=False, fix_map=False):
        """
        Return the Doppler imaging design matrix.

        This matrix dots into the spectral map to yield the model for the
        observed spectral timeseries (the ``flux``).

        Note that if this method is used to compute the spectral timeseries,
        the result should be reshaped into a matrix of shape
        (:py:attr:`nt`, :py:attr:`nw`) and optionally divided by the
        :py:meth:`baseline()` to match the return value of :py:meth:`flux()`.

        Args:
            theta (vector, optional): The angular phase(s) at which to compute
                the design matrix, in units of :py:attr:`angle_unit`. This
                must be a vector of size :py:attr:`nt`. Default is uniformly
                spaced values in the range ``[0, 2 * pi)``.
            fix_spectrum (bool, optional): If True, returns the design matrix
                for a fixed spectrum; this can then be dotted into the
                spherical harmonic coefficient matrix to obtain the flux. See
                below for details. Default is False.
            fix_map (bool, optional): If True, returns the design matrix
                for a fixed map; this can then be dotted into the spectrum
                matrix to obtain the flux. See below for details. Default is
                False.

        If ``fix_spectrum`` and ``fix_map`` are False (default), this method
        returns a sparse matrix of shape
        (:py:attr:`nt` * :py:attr:`nw`, :py:attr:`Ny` * :py:attr:`nw0_`).
        The flux may be computed from (assuming :py:attr:`lazy` is ``False``)

        .. code-block::python

            D = map.design_matrix()
            flux = D.dot(map.spectral_map).reshape(map.nt, map.nw)

        If ``fix_spectrum`` is True, returns a dense matrix of shape
        (:py:attr:`nt` * :py:attr:`nw`, :py:attr:`nc` * :py:attr:`Ny`).
        The flux may be computed from

        .. code-block::python

            D = map.design_matrix(fix_spectrum=True)
            y = map.y.transpose().flatten()
            flux = D.dot(y).reshape(map.nt, map.nw)

        Finally, if ``fix_map`` is True, returns a dense matrix of shape
        (:py:attr:`nt` * :py:attr:`nw`, :py:attr:`nc` * :py:attr:`nw0_`).
        The flux may be computed from

        .. code-block::python

            D = map.design_matrix(fix_map=True)
            spectrum = map.spectrum_.flatten()
            flux = D.dot(spectrum).reshape(map.nt, map.nw)

        .. note::

            Instantiating this matrix is usually a bad idea, since it can
            be very slow and consume a lot of memory. Check out the
            :py:meth:`dot` method instead for fast dot products with the design
            matrix.

        """
        theta = self._get_default_theta(theta)
        assert not (
            fix_spectrum and fix_map
        ), "Cannot fix both the spectrum and the map."

        # Compute the Doppler operator
        if fix_spectrum:

            # Fixed spectrum (dense)
            D = self.ops.get_D_fixed_spectrum(
                self._inc, theta, self._veq, self._u, self._spectrum
            )

        elif fix_map:

            # Fixed map (dense)
            D = self.ops.get_D_fixed_map(
                self._inc, theta, self._veq, self._u, self._y
            )

        else:

            # Full matrix (sparse)
            D = self.ops.get_D(self._inc, theta, self._veq, self._u)

        # Interpolate to the output grid
        if self._interp:
            D = self._math.sparse_dot(self._SBlock, D)

        return D

    def flux(self, theta=None, normalize=True, method="dotconv"):
        """
        Return the model for the full spectral timeseries.

        Args:
            theta (vector, optional): The angular phase(s) at which to compute
                the design matrix, in units of :py:attr:`angle_unit`. This
                must be a vector of size :py:attr:`nt`. Default is uniformly
                spaced values in the range ``[0, 2 * pi)``.
            normalize (bool, optional): Whether or not to normalize the flux.
                If True (default), normalizes the flux so that the continuum
                level is unity at all epochs. If False, preserves the natural
                changes in the continuum as the total flux of the star changes
                during its rotation. Note that this is not usually an
                observable effect, since spectrographs aren't designed to
                preserve this information!
            method (str, optional): The strategy for computing the flux. Must
                be one of ``dotconv``, ``convdot``, ``conv``, or ``design``.
                Default is ``dotconv``, which is the fastest method in most
                cases. All three of ``dotconv``, ``convdot``, and ``conv``
                compute the flux via fast two-dimensional convolutions
                (via ``conv2d``), while ``design`` computes the flux by
                instantiating the design matrix and dotting it in. This last
                method is usually extremely slow and memory intensive; its
                use is not recommended in general.

        This method returns a matrix of shape (:py:attr:`nt`, :py:attr:`nw`)
        corresponding to the model for the observed spectrum (evaluated on the
        wavelength grid :py:attr:`wav`) at each of :py:attr:`nt` epochs.

        """
        theta = self._get_default_theta(theta)
        if method == "dotconv":
            flux = self.ops.get_flux_from_dotconv(
                self._inc, theta, self._veq, self._u, self._y, self._spectrum
            )
        elif method == "convdot":
            flux = self.ops.get_flux_from_convdot(
                self._inc, theta, self._veq, self._u, self._y, self._spectrum
            )
        elif method == "conv":
            flux = self.ops.get_flux_from_conv(
                self._inc, theta, self._veq, self._u, self.spectral_map
            )
        elif method == "design":
            flux = self.ops.get_flux_from_design(
                self._inc, theta, self._veq, self._u, self.spectral_map
            )
        else:
            raise ValueError(
                "Keyword ``method`` must be one of ``dotconv``, "
                "``convdot``, ``conv``, or ``design``."
            )

        # Interpolate to the output grid
        if self._interp:
            flux = self._math.sparse_dot(flux, self._S)

        # Remove the baseline?
        if normalize:
            flux /= self._math.reshape(
                self.ops.get_baseline(
                    self._inc,
                    theta,
                    self._veq,
                    self._u,
                    self._continuum * self._y,
                ),
                (self.nt, 1),
            )

        return flux

    def baseline(self, theta=None):
        """
        Return the photometric baseline at each epoch.

        Args:
            theta (vector, optional): The angular phase(s) at which to compute
                the design matrix, in units of :py:attr:`angle_unit`. This
                must be a vector of size :py:attr:`nt`. Default is uniformly
                spaced values in the range ``[0, 2 * pi)``.
        """
        theta = self._get_default_theta(theta)
        baseline = self.ops.get_baseline(
            self._inc, theta, self._veq, self._u, self._continuum * self._y
        )
        return baseline

    def dot(
        self, x, theta=None, transpose=False, fix_spectrum=False, fix_map=False
    ):
        """
        Dot the Doppler design matrix into a given matrix or vector.

        This method is useful for computing dot products between the design
        matrix and the spectral map (to compute the model for the spectrum)
        or between the design matrix and (say) a covariance matrix (when doing
        inference). This is in general much, much faster than instantiating the
        :py:meth:`design_matrix` explicitly and dotting it in.

        Args:
            x (vector or matrix): The column vector or matrix into which
                the design matrix is dotted. This argument must have a specific
                shape that depends on the other arguments to this method;
                see below.
            theta (vector, optional): The angular phase(s) at which to compute
                the design matrix, in units of :py:attr:`angle_unit`. This
                must be a vector of size :py:attr:`nt`. Default is uniformly
                spaced values in the range ``[0, 2 * pi)``.
            transpose (bool, optional): If True, dots the transpose of the
                design matrix into ``x``. Default is False.
            fix_spectrum (bool, optional): If True, performs the operation
                using the design matrix for a fixed spectrum. The current
                spectrum is "baked into" the design matrix, so the tensor ``x``
                is a representation of the spherical harmonic decomposition of
                the surface. Default is False.
            fix_map (bool, optional): If True, performs the oprtation using the
                the design matrix for a fixed map. The current spherical
                harmonic map is "baked into" the design matrix, so the tensor
                ``x`` is a representation of the spectral decomposition of the
                surface. Default is False.

        The shapes of ``x`` and of the matrix returned by this method depend on
        the method's arguments. If ``transpose`` is False, this returns a dense
        matrix of shape (:py:attr:`nt` * :py:attr:`nw`, ``...``), where ``...``
        are any additional dimensions in ``x`` beyond the first. In this case,
        the shape of ``x`` should be as follows:

            - If ``fix_spectrum`` and ``fix_map`` are False (default), the
              input argument ``x`` must have shape
              (:py:attr:`Ny` * :py:attr:`nw0_`, ``...``).

            - If ``fix_spectrum`` is True, the input argument ``x`` must have
              shape (:py:attr:`nc` * :py:attr:`Ny`, ``...``).

            - If ``fix_map`` is True, the input argument ``x`` must have
              shape (:py:attr:`nc` * :py:attr:`nw0_`, ``...``).

        If, instead, ``transpose`` is True, this returns a dense
        matrix of a shape that depends on the arguments:

            - If ``fix_spectrum`` and ``fix_map`` are False (default), the
              return value has shape
              (:py:attr:`Ny` * :py:attr:`nw0_`, ``...``).

            - If ``fix_spectrum`` is True, the return value has shape
              (:py:attr:`nc` * :py:attr:`Ny`, ``...``).

            - If ``fix_map`` is True, the return value has shape
              (:py:attr:`nc` * :py:attr:`nw0_`, ``...``).

        When ``transpose`` is True, the input argument ``x`` must always have
        shape (:py:attr:`nt` * :py:attr:`nw`, ``...``).

        Note that if this method is used to compute the spectral timeseries
        (with ``tranpose = False``), the result should be reshaped into a
        matrix of shape (:py:attr:`nt`, :py:attr:`nw`) to match the return
        value of :py:meth:`flux()` and optionally divided by the
        :py:meth:`baseline()` to match the return value of :py:meth:`flux()`.

        """
        x = self._math.cast(x)
        theta = self._get_default_theta(theta)
        assert not (
            fix_spectrum and fix_map
        ), "Cannot fix both the spectrum and the map."

        if transpose:

            # Interpolate from `wav` to `wav_` at each epoch
            if self._interp:
                x = self._math.sparse_dot(self._STrBlock, x)

            if fix_spectrum:

                # This is inherently fast -- no need for a special Op
                D = self.ops.get_D_fixed_spectrum(
                    self._inc, theta, self._veq, self._u, self._spectrum
                )
                product = self._math.dot(self._math.transpose(D), x)

            elif fix_map:

                product = self.ops.dot_design_matrix_fixed_map_transpose_into(
                    self._inc, theta, self._veq, self._u, self._y, x
                )

            else:

                product = self.ops.dot_design_matrix_transpose_into(
                    self._inc, theta, self._veq, self._u, x
                )

        else:

            if fix_spectrum:

                # This is inherently fast -- no need for a special Op
                D = self.ops.get_D_fixed_spectrum(
                    self._inc, theta, self._veq, self._u, self._spectrum
                )
                product = self._math.dot(D, x)

            elif fix_map:

                product = self.ops.dot_design_matrix_fixed_map_into(
                    self._inc, theta, self._veq, self._u, self._y, x
                )

            else:

                product = self.ops.dot_design_matrix_into(
                    self._inc, theta, self._veq, self._u, x
                )

            # Interpolate from `wav_` to `wav` at each epoch
            if self._interp:
                product = self._math.sparse_dot(self._SBlock, product)

        return product

    def show(self, theta=None, res=150, file=None, **kwargs):
        """
        Display (or save) an interactive visualization of the star.

        This method uses the ``bokeh`` package to render an interactive
        visualization of the spectro-spatial stellar surface and the
        model for the spectral timeseries. The output is an HTML page that
        is either saved to disk (if ``file`` is provided) or displayed in
        a browser window or inline (if calling this method from within a
        Jupyter notebook).

        Users can interact with the visualization by moving the mouse over
        the map to show the emergent, rest frame spectrum at different points
        on the surface. Users can also scroll (with the mouse wheel or track
        pad) to change the wavelength at which the map is visualized (in the
        left panel) or to rotate the orthographic projection of the map (in
        the right panel).

        Args:
            theta (vector, optional): The angular phase(s) at which to compute
                the design matrix, in units of :py:attr:`angle_unit`. This
                must be a vector of size :py:attr:`nt`. Default is uniformly
                spaced values in the range ``[0, 2 * pi)``.
            res (int, optional): Resolution of the map image in pixels on a
                side. Default is ``150``.
            file (str, optional): Path to an HTML file to which the
                visualization will be saved. Default is None, in which case
                the visualization is displayed.

        .. note::

            The visualization can be somewhat memory-intensive! Try decreasing
            the map resolution if you experience issues.

        """
        with CompileLogMessage("show", custom_message="Rendering the map..."):

            get_val = evaluator(**kwargs)
            if theta is None:
                theta = np.linspace(0, 2 * np.pi, self.nt, endpoint=False)
            else:
                if is_tensor(theta):
                    theta = get_val(theta)
                theta *= self._angle_factor

            # Render the map
            moll = np.zeros((self.nc, res, res))
            ortho = np.zeros((self.nt, res, res))
            for k in range(self.nc):
                self._map[:, :] = self._y[:, k]
                img = get_val(self._map.render(projection="moll", res=res))
                moll[k] = img
                ortho += get_val(
                    self._spectrum[k, 0]
                    * self._map.render(
                        projection="ortho",
                        theta=theta / self._angle_factor,
                        res=res,
                    )
                )

            # Get the observed spectrum at each phase (vsini = 0)
            veq = self.veq
            self.veq = 0.0
            flux0 = get_val(
                self.flux(theta / self._angle_factor, normalize=True)
            )
            self.veq = veq

            # Get the observed spectrum at each phase
            flux = get_val(
                self.flux(theta / self._angle_factor, normalize=True)
            )

            # Init the web app
            viz = Visualize(
                get_val(self.wav0),
                get_val(self.wav),
                moll,
                ortho,
                get_val(self.spectrum),
                theta,
                flux0,
                flux,
                get_val(self._inc),
            )

        # Save or display
        viz.show(file=file)

    def solve(
        self,
        flux,
        flux_err=None,
        theta=None,
        spatial_mean=None,
        spatial_cov=None,
        spatial_inv_cov=None,
        spectral_mean=None,
        spectral_cov=None,
        normalized=True,
        baseline=None,
        fix_spectrum=False,
        fix_map=False,
    ):
        """
        Solve the linear problem for the spatial and/or spectral map
        given a spectral timeseries.

        .. warning::

            This method is still being developed!

        """
        # --------------------------
        # ---- Process defaults ----
        # --------------------------

        if flux_err is None:
            flux_err = 1e-6
        if spatial_mean is None:
            spatial_mean = 0.0
        if spatial_cov is None and spatial_inv_cov is None:
            spatial_cov = 1e-3

        # ----------------------
        # ---- Check shapes ----
        # ----------------------

        # Flux must be a matrix (nt, nw)
        if self.nt == 1:
            flux = self._math.reshape(
                self._math.cast(flux), (self.nt, self.nw)
            )
        else:
            flux = self.ops.enforce_shape(
                self._math.cast(flux), np.array([self.nt, self.nw])
            )

        # Flux error may be a scalar or a matrix (nt, nw)
        flux_err = self._math.cast(flux_err)
        if flux_err.ndim == 0:
            flux_err = flux_err * self._math.ones((self.nt, self.nw))
        else:
            if self.nt == 1:
                flux_err = self._math.reshape(flux_err, (self.nt, self.nw))
            else:
                flux_err = self.ops.enforce_shape(
                    flux_err, np.array([self.nt, self.nw])
                )

        # Spatial mean may be a scalar, a vector (Ny), or a list of those
        # Reshape it to a matrix of shape (Ny, nc)
        if type(spatial_mean) not in (list, tuple):
            # Use the same mean for all components
            spatial_mean = [spatial_mean for n in range(self.nc)]
        else:
            # Check that we have one mean per component
            assert len(spatial_mean) == self.nc
        for n in range(self.nc):
            spatial_mean[n] = self._math.cast(spatial_mean[n])
            assert spatial_mean[n].ndim < 2
            spatial_mean[n] = self._math.reshape(
                spatial_mean[n] * self._math.ones(self.Ny), (-1, 1)
            )
        spatial_mean = self._math.concatenate(spatial_mean, axis=-1)

        # Spatial (inv) cov may be a scalar, a vector, a matrix (Ny, Ny),
        # or a list of those. Invert it if needed and reshape to a matrix of
        # shape (Ny, nc) (inverse variances) or a tensor of shape
        # (Ny, Ny, nc) (nc separate inverse covariance matrices)
        if spatial_cov is not None:

            # User provided the *covariance*

            if type(spatial_cov) not in (list, tuple):
                # Use the same covariance for all components
                spatial_cov = [spatial_cov for n in range(self.nc)]
            else:
                # Check that we have one covariance per component
                assert len(spatial_cov) == self.nc
            spatial_inv_cov = [None for n in range(self.nc)]

            for n in range(self.nc):
                spatial_cov[n] = self._math.cast(spatial_cov[n])
                assert spatial_cov[n].ndim == spatial_cov[0].ndim
                if spatial_cov[n].ndim < 2:
                    spatial_inv_cov[n] = self._math.reshape(
                        self._math.ones(self.Ny) / spatial_cov[n], (-1, 1)
                    )
                else:
                    cho = self._math.cholesky(spatial_cov[n])
                    inv = self._linalg.cho_solve(cho, self._math.eye(self.Ny))
                    spatial_inv_cov[n] = self._math.reshape(
                        inv, (self.Ny, self.Ny, 1)
                    )
        else:

            # User provided the *inverse covariance*

            if type(spatial_inv_cov) not in (list, tuple):
                # Use the same covariance for all components
                spatial_inv_cov = [spatial_inv_cov for n in range(self.nc)]
            else:
                # Check that we have one covariance per component
                assert len(spatial_inv_cov) == self.nc

            for n in range(self.nc):
                spatial_inv_cov[n] = self._math.cast(spatial_inv_cov[n])
                assert spatial_inv_cov[n].ndim == spatial_inv_cov[0].ndim
                if spatial_inv_cov[n].ndim < 2:
                    spatial_inv_cov[n] = self._math.reshape(
                        self._math.ones(self.Ny) * spatial_inv_cov[n], (-1, 1)
                    )
                else:
                    spatial_inv_cov[n] = self._math.reshape(
                        spatial_inv_cov[n], (self.Ny, self.Ny, 1)
                    )

        # Tensor of nc inverse variance vectors or covariance matrices
        spatial_inv_cov = self._math.concatenate(spatial_inv_cov, axis=-1)

        # Baseline must be a vector (nt,)
        if baseline is not None:
            baseline = self.ops.enforce_shape(
                self._math.cast(baseline), np.array([self.nt])
            )

        # ----------------
        # ---- Solve! ----
        # ----------------

        if fix_spectrum:

            # The spectrum is fixed. We are solving the for spatial map.

            # Get the design matrix conditioned on the current spectrum
            D = self.design_matrix(theta=theta, fix_spectrum=True)

            if (not normalized) or (baseline is not None):

                # The problem is exactly linear!

                # De-normalize the data w/ the given baseline?
                if normalized:
                    f = flux * self._math.reshape(baseline, (self.nt, 1))
                    ferr = flux_err * self._math.reshape(
                        baseline, (self.nt, 1)
                    )
                else:
                    f = flux
                    ferr = flux_err

                # Unroll the data into a vector, and reshape the priors
                f = self._math.reshape(f, (-1,))
                ferr = self._math.reshape(ferr, (-1,))
                mu = self._math.reshape(
                    self._math.transpose(spatial_mean), (-1)
                )
                if spatial_inv_cov.ndim == 2:
                    invL = self._math.reshape(
                        self._math.transpose(spatial_inv_cov), (-1)
                    )
                else:
                    invL = self._math.block_diag(
                        *[spatial_inv_cov[:, :, n] for n in range(self.nc)]
                    )

                # Solve the L2 problem
                mean, cho_cov = self._linalg.solve(D, f, ferr, mu, invL)

                # Set the current map to the MAP
                self[:, :, :] = self._math.transpose(
                    self._math.reshape(mean, (self.nc, self.Ny))
                )

                # Return the factorized posterior covariance
                return cho_cov

            else:

                # TODO: Everything!

                # The problem is approximately linear in the limit
                # that the change in the photometric baseline over
                # time is small.

                # Get the baseline operator
                B = self.baseline_matrix(theta=theta)

                # Indices of Y_{0,0} coeffs
                i0 = np.zeros(map.nc * map.Ny, dtype=bool)
                i0[0 :: map.Ny] = 1

                # Split the operators into l = 0 and l > 0 terms
                # The problem is approximately linear in the l > 0 terms
                D0 = D[:, i0]
                D1 = D[:, ~i0]
                B0 = B[:, i0]
                B1 = B[:, ~i0]

                # TODO: User input
                y0 = self._math.ones(self.nc) / self.nc

                dot = self._math.dot
                reshape = self._math.reshape

                # The approximate linear problem is
                #
                #       flux ~ A @ y1 + b
                #
                D0y0 = dot(D0, y0)
                B0y0 = dot(B0, y0)
                b = 2 * D0y0 - D0y0 * B0y0
                A = (
                    2 * D1
                    - reshape(D0y0, (-1, 1)) * B1
                    - reshape(B0y0, (-1, 1)) * D1
                )

                y1 = np.linalg.solve(
                    A.T @ A / ferr ** 2 + I / siga ** 2,
                    A.T @ (f - b) / ferr ** 2,
                )

                y1, _ = solve(
                    A,
                    flux - b,
                    C=flux_err ** 2,
                    cho_C=None,
                    mu=spatial_mean,
                    L=spatial_cov,
                    cho_L=spatial_cho_cov,
                    N=self.nc * self.Ny,
                    lazy=self.lazy,
                )

                # The exact linear problem (conditioned on `y`) is
                #
                #       flux ~ (D1 @ y1 + b) / (B0 @ y0 + B1 @ y1)
                #
                y1 = np.linalg.solve(
                    D1.T @ D1 / ferr ** 2 + I / sigb ** 2,
                    D1.T @ (f * (B @ y) - b) / ferr ** 2,
                )
                y = np.zeros(map.nc * map.Ny)
                y[i0] = y0
                y[~i0] = y1

                # TODO
                raise NotImplementedError("Not yet implemented.")

            return cho_C

        else:

            # TODO
            raise NotImplementedError("Not yet implemented.")
