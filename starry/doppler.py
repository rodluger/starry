# -*- coding: utf-8 -*-
from . import config
from ._constants import *
from ._core import OpsDoppler, math
from ._core.utils import is_tensor, CompileLogMessage
from ._core.math import nadam
from ._indices import integers, get_ylm_inds, get_ylmw_inds, get_ul_inds
from .compat import evaluator, tt
from .maps import YlmBase, MapBase, Map
from .doppler_visualize import Visualize
from .doppler_solve import Solve
import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import csr_matrix
from warnings import warn
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy import units
from tqdm.auto import tqdm


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
        wavc (scalar, optional): A wavelength at which there are no spectral
            features in the *observed* spectrum. This is used for normalizing
            the model at each epoch. Default is the first element of
            :py:attr:`wav`.
        oversample (float, optional): The oversampling factor for the internal
            wavelength grid. The number of spectral bins used to perform the
            convolutions is equal to this value times the length of
            :py:attr:`wav`. If both, `wav0` and `wav` are provided, default is
            the ratio of the lengths of the two arrays. Otherwise,
            default is 1.
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
    _default_wav = np.linspace(642.75, 643.25, 200)  # FeI 6430

    def __init__(
        self,
        ydeg=1,
        udeg=0,
        nc=1,
        nt=10,
        wav=None,
        wav0=None,
        wavc=None,
        oversample=None,
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

        # Compute the index of the continuum (`wav` grid)
        if wavc is None:
            # The first element of `wav`
            self._continuum_idx = 0
        else:
            assert not is_tensor(
                wavc
            ), "The continuum wavelength must be a numerical quantity."
            # The closest element in `wav` to `wavc`
            self._continuum_idx = np.argmin(np.abs(wav - wavc))

        # Compute the size of the internal wavelength grid
        # Note that this must be odd!
        if wav0 is None:
            oversample = 1
        else:
            oversample = int(np.ceil(len(wav0) / len(wav)))
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
        hw = np.array(
            np.ceil(
                0.5
                * np.abs(np.log((1 + betasini_max) / (1 - betasini_max)))
                / dlam
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
                "Edge effects may occur. See the documentation for more details."
            )

        # Mask unused indices in the rest-frame spectrum?
        self._mask_unused_wavelength_bins = kwargs.pop(
            "mask_unused_wavelength_bins", True
        )
        self._wav0_padding_left = wav0 < wav0_int[0]
        self._wav0_extrapolate_left = np.argmax(wav0 > wav0_int[0])
        self._wav0_padding_right = wav0 > wav0_int[-1]
        self._wav0_extrapolate_right = (
            nw0 - np.argmax(wav0[::-1] < wav0_int[-1]) - 1
        )

        # Index of the continuum (`wav0` grid)
        self._continuum_idx0 = np.argmin(
            np.abs(wav0 - wav[self._continuum_idx])
        )

        # Interpolation between internal grid and user grid

        # TODO: Integrate over each wavelength bin in the output?
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

            # Interpolate from `wav_` to `wav`
            # These are used to interpolate the model and the design
            # matrix from the internal (i) onto the external (e) grid
            S = self._get_spline_operator(wav_int, wav)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._Si2eTr = self._math.sparse_cast(S.T)
            self._Si2eBlk = self._math.sparse_cast(
                sparse_block_diag([S for n in range(nt)], format="csr")
            )
            self._Si2eTrBlk = self._math.sparse_cast(
                sparse_block_diag([S.T for n in range(nt)], format="csr")
            )

            # Interpolate from `wav0_` to `wav0`
            S = self._get_spline_operator(wav0_int, wav0)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._S0i2e = self._math.sparse_cast(S)
            self._S0i2eTr = self._math.sparse_cast(S.T)

            # Interpolate from `wav` to `wav_`
            S = self._get_spline_operator(wav, wav_int)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._Se2i = self._math.sparse_cast(S)

            # Interpolate from `wav0` to `wav0_`
            S = self._get_spline_operator(wav0, wav0_int)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._S0e2i = self._math.sparse_cast(S)
            self._S0e2iTr = self._math.sparse_cast(S.T)

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

        # Linear solver
        self._solver = Solve(self)

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
        The rest frame spectrum for each component.

        This quantity is defined on the wavelength grid :py:attr:`wav0`.
        Shape must be (:py:attr:`nc`, :py:attr:`nw0`). If :py:attr:`nc` is
        unity, a one-dimensional array of length :py:attr:`nw0` is also
        accepted.

        """
        # Interpolate to the ``wav0`` grid.
        if self._interp:
            spectrum = self._math.sparse_dot(self._spectrum, self._S0i2eTr)
            if self._mask_unused_wavelength_bins:
                if self._lazy:
                    spectrum = tt.set_subtensor(
                        spectrum[:, self._wav0_padding_left],
                        tt.reshape(
                            spectrum[:, self._wav0_extrapolate_left], (-1, 1)
                        ),
                    )
                    spectrum = tt.set_subtensor(
                        spectrum[:, self._wav0_padding_right],
                        tt.reshape(
                            spectrum[:, self._wav0_extrapolate_right], (-1, 1)
                        ),
                    )
                else:
                    spectrum[:, self._wav0_padding_left] = spectrum[
                        :, self._wav0_extrapolate_left
                    ].reshape(-1, 1)
                    spectrum[:, self._wav0_padding_right] = spectrum[
                        :, self._wav0_extrapolate_right
                    ].reshape(-1, 1)
            return spectrum
        else:
            return self._spectrum

    @spectrum.setter
    def spectrum(self, spectrum):
        # Cast & reshape
        if self._nc == 1:
            spectrum = self._math.reshape(
                self._math.cast(spectrum), (1, self._nw0)
            )
        else:
            spectrum = self.ops.enforce_shape(
                self._math.cast(spectrum), np.array([self._nc, self._nw0])
            )

        # Interpolate from the ``wav0`` grid to internal, padded grid
        if self._interp:
            self._spectrum = self._math.sparse_dot(spectrum, self._S0e2iTr)
        else:
            self._spectrum = spectrum

    @property
    def spectrum_(self):
        """
        The *internal* rest frame spectrum for each component. *Read only*

        This is defined on the wavelength grid :py:attr:`wav0_`.

        """
        return self._spectrum

    @property
    def wavc(self):
        """
        A wavelength corresponding to the continuum level. *Read only*

        This is specified when instantiating a :py:class:`DopplerMap` and
        should be a wavelength at which there are no spectral
        features in the *observed* spectrum. This is used for normalizing
        the model at each epoch.

        """
        return self.wav[self._continuum_idx]

    @property
    def y(self):
        """The spherical harmonic coefficient matrix. *Read-only*"""
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
        elif isinstance(idx, tuple) and len(idx) == 2:
            # User is accessing a Ylm index
            inds = get_ylmw_inds(
                self.ydeg, self.nc, idx[0], idx[1], slice(None, None, None)
            )
            return self._y[inds]
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
        Set the spherical harmonic or limb darkening coefficient(s).

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
        elif isinstance(idx, tuple) and len(idx) == 2:
            # User is accessing a Ylm index
            i, j = get_ylmw_inds(
                self.ydeg, self.nc, idx[0], idx[1], slice(None, None, None)
            )
            self._y = self.ops.set_matrix(self._y, i, j, val)
        elif isinstance(idx, tuple) and len(idx) == 3:
            # User is accessing a Ylmc index
            i, j = get_ylmw_inds(self.ydeg, self.nc, idx[0], idx[1], idx[2])
            if self.nc == 1:
                assert np.array_equal(
                    j.reshape(-1), [0]
                ), "Invalid map component."
            self._y = self.ops.set_matrix(self._y, i, j, val)
        else:
            raise ValueError("Invalid map index.")

    def reset(self, **kwargs):
        """Reset all map coefficients and attributes."""
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
        self._spectrum = self._math.cast(np.ones((self.nc, self.nw0_)))

        # Basic properties
        self.inc = kwargs.pop("inc", 0.5 * np.pi / self._angle_factor)
        self.obl = kwargs.pop("obl", 0.0)
        self.veq = kwargs.pop("veq", 0.0)

    def _get_SHT_matrix(self, nlat, nlon, eps=1e-12, smoothing=None):
        """
        Return the SHT matrix for transforming a lat-lon intensity
        grid to a vector of spherical harmonic coefficients.

        This method is only used internally to load images on rectangular
        lat-lon grids in the ``load`` method.

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
        map=None,
        maps=None,
        spectrum=None,
        spectra=None,
        cube=None,
        smoothing=None,
        fac=1.0,
        eps=1e-12,
    ):
        """Load a spatial and/or spectral representation of a stellar surface.

        Users may load spatial ``maps`` and/or ``spectra``: one of each is
        required per map component (:py:attr:`nc`). Alternatively, users may
        provide a single data ``cube`` containing the full spectral-spatial
        representation of the surface. This cube should have shape
        (``nlat``, ``nlon``, :py:attr:`nw0`), where ``nlat`` is the number
        of pixels in latitude, ``nlon`` is the number of pixels in longitude,
        and :py:attr:`nw0` is the number of wavelength bins in the rest frame
        spectrum.

        This routine performs a simple spherical harmonic transform (SHT)
        to compute the spherical harmonic expansion given maps or a data
        cube. If a data cube is provided, this routine performs singular
        value decomposition (SVD) to compute the :py:attr:`nc` component
        surface maps and spectra (the "eigen" components) that best approximate
        the input.

        Args:
            map (str or ndarray, optional): A list or ``ndarray`` of
                surface maps on a rectangular latitude-longitude grid. This
                may also be a list of strings corresponding to the names of
                PNG image files representing the surface maps.
            maps (str or ndarray, optional): Alias for ``map``.
            spectrum (ndarray, optional): A list or ``ndarray`` of vectors
                containing the spectra corresponding to each map component.
            spectra (ndarray, optional): Aliass for ``spectrum``.
            cube (ndarray, optional): A 3-dimensional ``ndarray`` of shape
                (``nlat``, ``nlon``, :py:attr:`nw0`) containing the spectra
                at each position on a latitude-longitude grid spanning the
                entire surface.
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
        # Aliases
        if maps is None:
            maps = map
        if spectra is None:
            spectra = spectrum

        if maps is not None or spectra is not None:

            # Input checks
            assert (
                cube is None
            ), "Cannot specify (`maps` or `spectra`) and `cube` simultaneously."

            if maps is not None:

                # ------------------------------
                # User is loading spatial maps
                # ------------------------------

                # Input checks
                if type(maps) is str:
                    assert self.nc == 1, "Must provide one map per component."
                    maps = [maps]
                elif type(maps) in (tuple, list):
                    maps = list(maps)
                    assert (
                        len(maps) == self.nc
                    ), "Must provide one map per component."
                elif type(maps) is np.ndarray:
                    if self.nc == 1:
                        if maps.ndim == 2:
                            maps = np.array([maps])
                    assert (
                        maps.shape[0] == self.nc
                    ), "Must provide one map per component."
                else:
                    raise TypeError("Invalid type for `maps`.")

                # Process each map
                Q = np.empty((self.Ny, 0))
                y = np.zeros((self.Ny, self.nc))
                y[:, 0] = 1.0
                for n, image in enumerate(maps):

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

                        raise TypeError("Invalid type for one of the `maps`.")

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
                        spectra, self._S0e2iTr
                    )
                else:
                    self._spectrum = spectra

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
                self._spectrum = self._math.sparse_dot(
                    self._math.cast(VT), self._S0e2iTr
                )
            else:
                self._spectrum = self._math.cast(VT)

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

    def _get_spline_operator(self, input_grid, output_grid):
        """ """
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
        """ """
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

    def sht_matrix(
        self,
        inverse=False,
        return_grid=False,
        smoothing=None,
        oversample=2,
        lam=1e-6,
    ):
        """
        Return the Spherical Harmonic Transform (SHT) matrix.

        This matrix dots into a vector of pixel intensities defined on
        a set of latitude-longitude points, resulting in a vector of spherical
        harmonic coefficients that best approximates the image.

        This can be useful for transforming between the spherical harmonic and
        pixel representations of the surface map, such as when specifying
        priors during inference or optimization. A common prior in Doppler
        imaging problems is the assumption of sparsity, often in the form of
        maximum entropy regularization, where the likelihood penalty has the
        form (e.g., Vogt et al. 1987)

            .. code-block::python

                S = -np.sum(np.log(p) * p)

        In this case, the prior must be applied to the vector ``p`` of
        *pixel intensities*, not the vector of spherical harmonic coefficients.
        The former is obtained from the latter as follows:

            .. code-block::python

                A = map.sht_matrix(inverse=True)
                p = np.dot(A, map.y)

        (Note, importantly, that ``p`` is not guaranteed to be positive, so
        one should be careful when taking the log in the expression above!)

        An alternative way to approach the problem is to sample in pixels ``p``
        directly; in this case, one must compute ``y`` from ``p`` so that
        ``starry`` can compute the model for the flux:

            .. code-block::python

                A = map.sht_matrix()
                y = np.dot(A, p)
                map[:, :] = y

        Args:
            inverse (bool, optional). If True, returns the inverse transform,
                which transforms from spherical harmonic coefficients to
                pixels. Default is False.
            return_grid (bool, optional). If True, also returns an array of
                shape ``(npix, 2)`` corresponding to the latitude-longitude
                points (in units of :py:attr:`angle_unit`) on which the SHT is
                evaluated. Default is False.
            smoothing (float, optional): Gaussian smoothing strength.
                Increase this value to suppress ringing (forward SHT only) or
                explicitly set to zero to disable smoothing. Default is
                ``2/self.ydeg``.
            oversample (int, optional): Factor by which to oversample the
                pixelization grid. Default `2`.
            lam (float, optional): Regularization parameter for the inverse
                pixel transform. Default `1e-6`.

        Returns:
            A matrix of shape (:py:attr:`Ny`, ``npix``) or
            (:py:attr:`npix`, ``Ny``) (if ``inverse`` is True), where ``npix``
            is the number of pixels on the grid. This number is determined
            from the degree of the map and the ``oversample`` keyword. If
            ``return_grid`` is True, also returns the latitude-longitude points
            (in units of :py:attr:`angle_unit`) on which the SHT is
            evaluated, a matrix of shape ``(npix, 2)``.

        """
        lat, lon, ISHT, SHT, _, _ = self._map.get_pixel_transforms(
            oversample=oversample, lam=lam
        )
        if inverse:
            matrix = ISHT
        else:
            matrix = SHT
            if smoothing is None:
                smoothing = 2.0 / self.ydeg
            if smoothing > 0:
                l = np.concatenate(
                    [np.repeat(l, 2 * l + 1) for l in range(self.ydeg + 1)]
                )
                s = np.exp(-0.5 * l * (l + 1) * smoothing ** 2)
                matrix *= s[:, None]
        if return_grid:
            grid = np.vstack((lat, lon)).T
            return matrix, grid
        else:
            return matrix

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
            D = self._math.sparse_dot(self._Si2eBlk, D)

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
            flux = self._math.sparse_dot(flux, self._Si2eTr)

        # Remove the baseline?
        if normalize:
            flux /= self._math.reshape(
                flux[:, self._continuum_idx], (self.nt, 1)
            )

        return flux

    def baseline(self, theta=None, full=False):
        """
        Return the photometric baseline at each epoch.

        Args:
            theta (vector, optional): The angular phase(s) at which to compute
                the design matrix, in units of :py:attr:`angle_unit`. This
                must be a vector of size :py:attr:`nt`. Default is uniformly
                spaced values in the range ``[0, 2 * pi)``.
            full (bool, optional): If True, returns a matrix with the same
                shape as :py:meth:`flux`. If False (default), returns a vector
                of length :py:attr:`nc`.
        """
        # Get the design matrix for the continuum normalization
        C = self._math.reshape(
            self.design_matrix(theta=theta, fix_spectrum=True),
            [self.nt, self.nw, -1],
        )[:, self._continuum_idx, :]
        baseline = self._math.dot(
            C, self._math.reshape(self._math.transpose(self._y), [-1])
        )
        if full:
            baseline = self._math.repeat(baseline, self.nw, axis=0)
            baseline = self._math.reshape(baseline, (self.nt, self.nw))
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
                x = self._math.sparse_dot(self._Si2eTrBlk, x)

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
                product = self._math.sparse_dot(self._Si2eBlk, product)

        return product

    def _visualize_bokeh(self, theta=None, res=150, file=None, **kwargs):
        """
        Visualize the map using bokeh.

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
                    # Evaluate the ortho maps at the *continuum*
                    self.spectrum[k, self._continuum_idx0]
                    * self._map.render(
                        projection="ortho",
                        theta=theta / self._angle_factor,
                        res=res,
                    )
                )

            # Get the observed spectrum at each phase (vsini = 0)
            # TODO: If veq is too small, the convolution kernel is
            # too narrow and we run into discretization error, which
            # causes severe edge effects.
            veq = self.veq
            self.veq = 500.0
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

    def _visualize_matplotlib(
        self,
        show_maps=True,
        show_spectra=True,
        file=None,
        projection="moll",
        vmin=None,
        vmax=None,
        nmax=5,
        theta=0.0,
        **kwargs,
    ):
        """
        Show the individual component surface maps and spectra.

        """
        # If we're in lazy mode, we need to evaluate stuff
        get_val = evaluator(**kwargs)

        # Show at most `nmax` components
        nc = min(nmax, self.nc)
        nrows = int(show_maps) + int(show_spectra)
        assert (
            nrows > 0
        ), "At least one of `show_maps` or `show_spectra` must be True."
        fig, ax = plt.subplots(
            nrows, nc, figsize=(4 * nc + 0.5 + 1.5 * (nc > 1), 2 * nrows)
        )
        ax = np.reshape(ax, (nrows, nc))

        # Figure out normalization
        find_vmin = vmin is None
        find_vmax = vmax is None
        if show_maps:
            if find_vmin or find_vmax:
                if find_vmin:
                    vmin = np.inf
                if find_vmax:
                    vmax = -np.inf
                for n in range(nc):
                    self._map[:, :] = self._y[:, n]
                    img = get_val(
                        self._map.render(
                            theta=theta, projection=projection, res=50
                        )
                    )
                    if find_vmin:
                        vmin = min(np.nanmin(img), vmin)
                    if find_vmax:
                        vmax = max(np.nanmax(img), vmax)
            norm = Normalize(vmin=vmin, vmax=vmax)
        if show_spectra:
            wav0 = get_val(self.wav0)
            spectrum = get_val(self.spectrum)
            smin = np.nanmin(spectrum)
            smax = np.nanmax(spectrum)
            rng = max(1e-3, smax - smin)
            spad = 0.1 * rng

        # Plot
        for n in range(nc):
            i = 0
            if show_maps:
                self._map[:, :] = self._y[:, n]
                self._map.show(
                    ax=ax[i, n],
                    theta=theta,
                    projection=projection,
                    norm=norm,
                    **kwargs,
                )
                ax[i, n].set_aspect("auto")
                i += 1
            if show_spectra:
                ax[i, n].plot(wav0, spectrum[n])
                ax[i, n].set_ylim(smin - spad, smax + spad)
                ax[i, n].set_xlim(wav0[0], wav0[-1])
                if n == 0:
                    ax[i, n].set_ylabel("intensity", fontsize=10)
                else:
                    ax[i, n].set_yticklabels([])
                ax[i, n].set_xlabel("wavelength [nm]", fontsize=10)
                for tick in (
                    ax[i, n].xaxis.get_major_ticks()
                    + ax[i, n].yaxis.get_major_ticks()
                ):
                    tick.label.set_fontsize(8)

        # Colorbar
        if show_maps:
            cbar = fig.colorbar(ax[0, 0].images[0], ax=ax[0].ravel().tolist())
            cbar.ax.tick_params(labelsize=10)

            # Dummy colorbar for spacing
            if show_spectra:
                fig.colorbar(
                    ax[0, 0].images[0], ax=ax[1].ravel().tolist()
                ).ax.set_visible(False)

        # Show or save
        if file is None:
            plt.show()
        else:
            fig.savefig(file, bbox_inches="tight")
            plt.close()

    def visualize(self, backend="bokeh", **kwargs):
        """
        Display or save a visualization of the star with optional interactivity.

        If ``backend`` is set to ``bokeh``, this method uses the ``bokeh``
        package to render an interactive visualization of the
        spectro-spatial stellar surface and the model for
        the spectral timeseries. The output is an HTML page that
        is either saved to disk (if ``file`` is provided) or displayed in
        a browser window or inline (if calling this method from within a
        Jupyter notebook).

        Users can interact with the visualization by moving the mouse over
        the map to show the emergent, rest frame spectrum at different points
        on the surface. Users can also scroll (with the mouse wheel or track
        pad) to change the wavelength at which the map is visualized (in the
        left panel) or to rotate the orthographic projection of the map (in
        the right panel).

        If instead ``backend`` is set to ``matplotlib``, this method shows
        static plots of the component surface maps and/or spectra.

        Args:
            backend (str, optional): The visualization backend. Options are
                ``bokeh`` or ``matplotlib``. Default is ``bokeh``.
            theta (vector, optional): The angular phase(s) at which to compute
                the design matrix, in units of :py:attr:`angle_unit`. If
                ``backend`` is ``bokeh``, this must be a vector of size
                :py:attr:`nt`. If ``backend` is ``matplotlib``, this may be
                a scalar. Default is uniformly spaced values in the range
                ``[0, 2 * pi)`` (``bokeh``) or ``0.0`` (``matplotlib``).
            res (int, optional): Resolution of the map image in pixels on a
                side. Default is ``150``.
            file (str, optional): Path to an HTML file (``bokeh`` backend) or
                PDF, JPG, PNG, etc. figure file (``matplotlib`` backend) to
                which the visualization will be saved. Default is None, in
                which case the visualization is displayed.

        If the backend is set to ``matplotlib``, additional keyword arguments
        are allowed:

        Args:
            show_maps (bool, optional): Show the component surface maps?
                Default is True.
            show_spectra (bool, optional): Show the component spectra? Default
                is True.
            projection (str, optional): Cartographic projection to plot.
                Options are ``rect`` (equirectangular), ``moll`` (Mollweide),
                and ``ortho`` (orthographic). Default is ``moll``.
            vmin (float, optional): Minimum value in the color scale. Default
                is None.
            vmax (float, optional): Maximum value in the color scale. Default
                is None.
            nmax (int, optional): Maximum number of components to show. Default
                is 5.

        Any other keywords accepted by the ``show`` method are also allowed.

        .. note::

            The ``bokeh`` visualization can be somewhat memory-intensive!
            Try decreasing the map resolution or switch to the ``matplotlib``
            backend if you experience issues.

        """
        if backend == "bokeh":
            return self._visualize_bokeh(**kwargs)
        elif backend == "matplotlib":
            return self._visualize_matplotlib(**kwargs)
        else:
            raise ValueError(
                "Invalid backend. Options are ``bokeh`` or ``matplotlib``."
            )

    def show(self, n=0, **kwargs):
        """
        Show a component surface map. See the documentation for
        ``starry.Map().show()`` for details.

        Args:
            n (int, optional): The index of the surface map component. Default
                is ``0``.

        """
        self._map[:, :] = self._y[:, n]
        return self._map.show(**kwargs)

    def render(self, n=0, **kwargs):
        """
        Render a component surface map. See the documentation for
        ``starry.Map().render()`` for details.

        Args:
            n (int, optional): The index of the surface map component. Default
                is ``0``.

        """
        self._map[:, :] = self._y[:, n]
        return self._map.render(**kwargs)

    def intensity(self, n=0, **kwargs):
        """
        Return the intensity of a surface map component. See the documentation
        for ``starry.Map().intensity()`` for details.

        Args:
            n (int, optional): The index of the surface map component. Default
                is ``0``.

        """
        self._map[:, :] = self._y[:, n]
        return self._map.intensity(**kwargs)

    def solve(self, flux, solver="bilinear", **kwargs):
        """
        Iteratively solves the bilinear or nonlinear problem for the spatial
        and/or spectral map given a spectral timeseries.

        Args:
            flux (matrix): The observed spectral timeseries. Must be an ndarray
                of shape (py:attr:`nt`, py:attr:`nw`).
            solver (str, optional): Which solver to use. Options are "bilinear"
                or "nonlinear". Default is "bilinear".
            flux_err (float, vector, or matrix, optional): The data
                uncertainty. If a scalar, the data is assumed to be
                homoscedastic. If a vector, the data for each epoch is assumed
                to be homoscedastic; in this case, must have length
                py:attr:`nt`. If a matrix, must have shape
                (py:attr:`nt`, py:attr:`nw`). Note that correlated errors
                (i.e., non-diagonal data covariances) are not currently
                supported. Please reach out if this is something you'd like
                to see implemented. Default is `1e-4`.
            theta (vector, optional): The angular phase(s) at which the spectra
                were observed, in units of :py:attr:`angle_unit`. This
                must be a vector of size :py:attr:`nt`. Default is uniformly
                spaced values in the range ``[0, 2 * pi)``.
            spatial_mean (float, vector, or list, optional): The prior mean on
                the spherical harmonic coefficients of the map. If a scalar,
                the same mean is assumed for all coefficients. If a vector of
                length :py:attr:`Ny`, the same mean vector is assumed for all
                map components. Users can also provide a list of length
                :py:attr:`nc` containing scalars or vectors corresponding to
                the prior mean for each component. Default is the vector
                `[1.0, 0.0, ..., 0.0]`.
            spatial_cov (float, vector, matrix, or list, optional): The prior
                (co)variance on the spherical harmonic coefficients of the map.
                If a scalar, assumes the same variance for all map coefficients
                (with no covariance across them). If a vector of length
                :py:attr:`Ny`, sets the prior variance for each spherical
                harmonic coefficient individually and assumes the same variance
                vector for all map components. If a matrix of shape
                (:py:attr:`Ny`, :py:attr:`Ny`), sets the prior covariance of
                each map component equal to this matrix. Users can also provide
                a list of length :py:attr:`nc` containing scalars, vectors, or
                matrices corresponding to the prior covariance for each
                component. Default is `1e-4`.
            spectral_mean (float, vector, or list, optional): The prior mean
                on the spectral components of the map. If a scalar,
                the same mean is assumed for all spectral elements. If a vector
                of length :py:attr:`nw0`, the same mean vector is assumed for
                all spectral components. Users can also provide a list of
                length :py:attr:`nc` containing scalars or vectors
                corresponding to the prior mean for each component.
                Default is `1.0`.
            spectral_cov (float, vector, matrix, or list, optional): The prior
                (co)variance on the spectra. If a scalar, assumes the same
                variance for all spectral elements
                (with no covariance across them). If a vector of length
                :py:attr:`nw0`, sets the prior variance for each spectral
                element individually and assumes the same variance
                vector for all spectral components. If a matrix of shape
                (:py:attr:`nw0`, :py:attr:`nw0`), sets the prior covariance of
                each spectral component equal to this matrix. Users can also
                provide a list of length :py:attr:`nc` containing scalars,
                vectors, or matrices corresponding to the prior covariance for
                each component. Default is `1e-3`.
            spectral_guess (float, vector, or list, optional): The guess
                for the spectral components of the map. If a scalar,
                the same value is assumed for all spectral elements. If a vector
                of length :py:attr:`nw0`, the same guess vector is assumed for
                all spectral components. Users can also provide a list of
                length :py:attr:`nc` containing scalars or vectors
                corresponding to the guess for each component.
                Default is to compute the guess based on a deconvolution of the
                mean observed spectrum.
            spectral_lambda (float, optional): The regularization parameter for
                the L1 solver. Increasing this value increases the sparsity of
                the solution. Default is `1e5`.
            spectral_maxiter (int, optional): Maximum number of iterations in
                the L1 solver. Default is `100`.
            spectral_eps (float, optional): Small parameter added to the
                diagonal of the spectral covariance matrix for stability.
                Default is `1e-12`.
            spectral_tol (float, optional): Tolerance for termination of the
                L1 iterative solver. Default is `1e-8`.
            spectral_method (str, optional): Regularization method when solving
                for the spectrum. Options are "L1" or "L2" (default). When
                solving for both the spectrum and the map, the L1 solver is
                used to obtain an initial guess for the spectrum, regardless
                of this setting.
            normalized (bool, optional): Whether the ``flux`` dataset is
                continuum-normalized. Default is True. If it is normalized, the
                solution for the map is non-linear, but typically converges
                with an iterative tempered solver. See the ``nlogT`` tempering
                parameter below.
            baseline (vector, optional): If ``normalized`` is True, users may
                provide a vector of length :py:attr:`nt` corresponding to the
                photometric baseline at each epoch. This is used to
                un-normalize the data, restoring the linearity of the problem.
                If this quantity is not known exactly (as is almost always
                the case), do not provide a value for it; instead, set the
                ``baseline_var`` parameter, which controls the variance of the
                prior on the baseline; the code will then marginalize over it.
                Default is None.
            baseline_var (float, optional): Variance of the prior on the
                baseline, when ``baseline`` is not provided and the data is
                normalized. Default is `1e-2`.
            fix_spectrum (bool, optional): If True, fixes the spectrum at the
                current value and solves only for the map. Default is False.
            fix_map (bool, optional): If True, fixes the map at the
                current value and solves only for the spectrum. Default is
                False.
            logT0 (float, optional): Initial log temperature for the tempering
                scheme. Default is `12.0`. See the ``nlogT`` tempering
                parameter below for more details.
            logTf (float, optional): Final log temperature for the tempering
                scheme. Default is `0.0` (untempered). See the ``nlogT``
                tempering parameter below for more details.
            nlogT (int, optional): Number of steps in the tempering scheme.
                Default is `50`. The tempering scheme is used to solve for the
                map when the baseline is not known, or to solve for both the
                map and the specturm when neither are known. At each step,
                the square of the flux uncertainty (i.e., the variance) is
                multiplied by the current value of the "temperature", which
                is slowly decreased from an initial large number controlled
                by ``logT0`` to a small number, controlled by ``logTf``. In
                practice, the correct tempering schedule can greatly improve
                the odds that the solver converges to the global minimum.
                Different problems in general require different settings for
                these parameters, so fine tuning is usually required.
            quiet (bool, optional): Suppress messages and progress bars?
                Default is False.

        """
        if self.lazy:

            # We need to ensure all the inputs are
            # numerical quantities.
            get_val = evaluator(**kwargs)
            flux = get_val(flux)
            spectrum_ = get_val(self.spectrum_)
            y = get_val(self._y)
            u = get_val(self._u)
            veq = get_val(self._veq)
            inc = get_val(self._inc)  # rad
            theta = get_val(
                self._get_default_theta(kwargs.pop("theta", None))
            )  # rad
            for key in kwargs.keys():
                if key not in ["point", "model"]:
                    kwargs[key] = get_val(kwargs[key])

        else:

            y = self._y
            u = self._u
            spectrum_ = self.spectrum_
            veq = self._veq
            inc = self._inc  # rad
            theta = self._get_default_theta(kwargs.pop("theta", None))  # rad

        # Run the solver
        if solver.lower().startswith("bi"):
            soln = self._solver.solve_bilinear(
                flux, theta, y, spectrum_, veq, inc, u, **kwargs
            )
        elif solver.lower().startswith("non"):
            soln = self._solver.solve_nonlinear(
                flux, theta, y, spectrum_, veq, inc, u, **kwargs
            )
        else:
            raise ValueError("Invalid `solver`.")

        # Set map props
        self._y = soln["y"]
        self._spectrum = soln["spectrum_"]

        return soln

    def optimize(
        self,
        model=None,
        start=None,
        niter=10000,
        lr=1e-4,
        quiet=False,
        **kwargs,
    ):
        """
        Optimizes the log likelihood function within a `pymc3` model context
        using the ``Adam`` solver.

        Args:
            model (optional): The `pymc3` model to optimize. Default is None,
                in which case the current model context is used.
            start (optional): The starting point for the optimization. Default
                is ``model.test_point``.
            niter (int, optional): The number of iterations. Default is
                ``10000``.
            lr (float, optional): The learning rate. Default is ``1e-4``.
            quiet (bool, optional): If True, disables the progress bar. Default
                is False.
            kwargs: Additional kwargs passed directly to
                ``pymc3_ext.optim.Adam``.

        Below is an example of how to use this method.

        .. code-block:: python

            import pymc3 as pm
            import starry

            # Define a `pymc3` model
            with pm.Model() as model:

                # Instantiate a Doppler map
                # Assuming ``nt=16`` spectral epochs
                map = starry.DopplerMap(ydeg=15, nt=16)

                # SHT matrix: converts from pixels to Ylms
                A = map.sht_matrix(smoothing=0.075)
                npix = A.shape[1]

                # Prior on the map: intensity uniform in [0, 1]
                p = pm.Uniform("pixels", lower=0.0, upper=1.0, shape=(npix,))
                amp = pm.Uniform("amp", lower=0.0, upper=1.0)
                map[:, :] = amp * tt.dot(A, p)

                # Prior on the spectrum: Gaussian about unity
                map.spectrum = pm.Normal(
                    "spectrum", mu=1.0, sigma=1e-1, shape=(map.nw0,)
                )

                # Compute the model
                # Assuming `theta` is a vector of 16 rotational phases at
                # which each of the spectra were observed
                flux_model = map.flux(theta=theta)

                # Likelihood term
                # Assuming `flux` is the data and `flux_err` is the
                # (scalar) uncertainty
                pm.Normal(
                    "obs",
                    mu=tt.reshape(flux_model, (-1,)),
                    sd=flux_err,
                    observed=flux.reshape(-1,)
                )

                # Optimize (this function)
                map_soln, loss = map.optimize()

        """
        import pymc3 as pm
        import pymc3_ext as pmx

        # Get the model
        try:
            if model is None:
                model = pm.Model.get_context()
        except TypeError:
            raise ValueError(
                "This method must be run within a `pymc3` model context."
            )

        # Get the starting point
        if start is None:
            start = model.test_point

        # Iterate
        loss = []
        map_soln = None
        best_loss = np.inf
        for obj, point in tqdm(
            pmx.optim.optimize_iterator(
                pmx.optim.Adam(lr=lr, **kwargs),
                niter,
                start=start,
                model=model,
            ),
            total=niter,
            disable=quiet,
        ):
            loss.append(obj)
            if obj < best_loss:
                best_loss = obj
                map_soln = point

        return map_soln, np.array(loss)
