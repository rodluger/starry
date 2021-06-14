# -*- coding: utf-8 -*-
from . import config
from ._constants import *
from ._core import OpsDoppler, math
from ._core.utils import is_tensor, CompileLogMessage
from ._indices import integers, get_ylm_inds, get_ylmw_inds, get_ul_inds
from .compat import evaluator
from .maps import YlmBase, MapBase, Map
from .doppler_visualize import Visualize
from .linalg import solve, lnlike
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import csr_matrix
from warnings import warn


class Amplitude(object):
    def __get__(self, instance, owner):
        return instance._math.squeeze(instance._amp)

    def __set__(self, instance, value):
        instance._amp = instance._math.reshape(
            instance._math.cast(np.ones(instance.nc) * value), (1, instance.nc)
        )


class DopplerMap:
    """
    Class representing a spectral-spatial stellar surface.

    Instances of this class can be used to model the spatial and wavelength
    dependence of stellar surfaces and the spectral timeseries one would
    observe in the presence of rotation-induced Doppler shifts. This class
    is specifically suited to Doppler imaging studies. It is similar (but
    different in important ways) to the traditional ``starry.Map`` class.
    Currently, this class does not model occultations.

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
        spectrum (matrix, optional): The spectrum of the star. This should be
            a matrix of shape (:py:attr:`nc`, :py:attr:`nw0`), i.e., one
            spectrum per map component. Note that the spectrum should be
            normalized such that the continuum is unity for all spectral
            components. If a different weighting is desired for different
            components, change :py:attr:`amp` instead. Default is a single
            Gaussian absorption line at the central wavelength of the first
            spectral component, and unity for all other components.
        amp (scalar or vector, optional): The amplitude of each of the spectral
            components. Default is unity for all components.
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

        This consists of a single Gaussian absorption line at the central
        wavelength of the first spectral component, and unity for all
        other components.
        """
        return self._math.concatenate(
            (
                self._math.reshape(
                    1
                    - 0.5
                    * self._math.exp(
                        -0.5 * (self._wav0 - self._wavr) ** 2 / 0.05 ** 2
                    ),
                    (1, self._nw0),
                ),
                self._math.ones((self._nc - 1, self._nw0)),
            ),
            axis=0,
        )

    # The map amplitude
    amp = Amplitude()

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
        **kwargs
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
            wav = self._default_wav
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
            **kwargs
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
        # Interpolate to the ``wav0`` grid
        if self._interp:
            return self._math.sparse_dot(self._spectrum, self._S0)
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

        # Interpolate from ``wav0`` grid to internal, padded grid
        if self._interp:
            self._spectrum = self._math.sparse_dot(spectrum, self._S0Inv)
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
    def y(self):
        """The spherical harmonic coefficient matrix. *Read-only*

        Changing the spatial representation of the map should be done by
        directly indexing the instance of this class, as follows.

        If ``nc = 1``, index the map directly using two indices:
        ``map[l, m] = ...`` where ``l`` is the spherical harmonic degree and
        ``m`` is the spherical harmonic order. These may be integers or
        arrays of integers. Slice notation may also be used.

        If ``nc > 1``, index the map directly using three indices instead:
        ``map[l, m, c] = ...`` where ``c`` is the index of the map component.
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
        The spectral-spatial map vector.

        This is equal to the (unrolled) outer product of the spherical harmonic
        decompositions and their corresponding spectral components, weighted
        by their respective amplitudes. Dot the design matrix into this
        quantity to obtain the observed spectral timeseries (the
        :py:meth:`flux`).
        """
        # Outer product with the map
        return self._math.reshape(
            self._math.dot(self._amp * self._y, self._spectrum), (-1,)
        )

    def __getitem__(self, idx):
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
        if not is_tensor(val):
            val = np.array(val)
        if isinstance(idx, integers) or isinstance(idx, slice):
            # User is accessing a limb darkening index
            inds = get_ul_inds(self.udeg, idx)
            if 0 in inds:
                raise ValueError("The u_0 coefficient cannot be set.")
            self._u = self.ops.set_vector(self._u, inds, val)
            self._map._u = self._u
        elif isinstance(idx, tuple) and len(idx) == 2 and self.nc == 1:
            # User is accessing a Ylm index
            inds = get_ylm_inds(self.ydeg, idx[0], idx[1])
            if 0 in inds:
                if np.array_equal(np.sort(inds), np.arange(self.Ny)):
                    # The user is setting *all* coefficients, so we allow
                    # them to "set" the Y_{0,0} coefficient...
                    self._y = self._math.reshape(
                        self.ops.set_vector(self._y[:, 0], inds, val), (-1, 1)
                    )
                    # ... except we scale the amplitude of the map and
                    # force Y_{0,0} to be unity.
                    self._amp = self._math.reshape(self._y[0], (1, -1))
                    self._y /= self._y[0]
                else:
                    raise ValueError(
                        "The Y_{0,0} coefficient cannot be set. "
                        "Please change the map amplitude instead."
                    )
            else:
                self._y = self.ops.set_vector(self._y[:, 0], inds, val)
        elif isinstance(idx, tuple) and len(idx) == 3:
            # User is accessing a Ylmc index
            i, j = get_ylmw_inds(self.ydeg, self.nc, idx[0], idx[1], idx[2])
            if self.nc == 1:
                assert np.array_equal(
                    j.reshape(-1), [0]
                ), "Invalid map component."
            if 0 in i:
                if np.array_equal(np.sort(i.reshape(-1)), np.arange(self.Ny)):
                    # The user is setting *all* coefficients, so we allow
                    # them to "set" the Y_{0,0} coefficient...
                    self._y = self.ops.set_matrix(self._y, i, j, val)
                    # ... except we scale the amplitude of the map and
                    # force Y_{0,0} to be unity.
                    self._amp = self._math.reshape(
                        self.ops.set_vector(
                            self._math.reshape(self._amp, (self.nc,)),
                            j,
                            self._y[0, j],
                        ),
                        (1, self.nc),
                    )
                    self._y /= self._y[0]
                else:
                    raise ValueError(
                        "The Y_{0,0} coefficient cannot be set. "
                        "Please change the map amplitude instead."
                    )
            else:
                self._y = self.ops.set_matrix(self._y, i, j, val)
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
        self._amp = self._math.reshape(
            self._math.cast(np.ones(self.nc) * kwargs.pop("amp", 1.0)),
            (1, self.nc),
        )

        # Reset the spectrum
        self.spectrum = kwargs.pop("spectrum", self._default_spectrum())

        # Basic properties
        self.inc = kwargs.pop("inc", 0.5 * np.pi / self._angle_factor)
        self.obl = kwargs.pop("obl", 0.0)
        self.veq = kwargs.pop("veq", 0.0)

    def load(self, images, fix_amp=True, **kwargs):
        """
        Load a sequence of ndarrays or a series of images.

        This routine performs a simple spherical harmonic transform (SHT)
        to compute the spherical harmonic expansion corresponding to
        a list of input image files or a sequencey of ``numpy`` ndarrays on a
        lat-lon grid. The resulting coefficients are ingested into the map.

        Args:
            images: A list of paths to PNG files or a sequence of two-dimensional
                ``numpy`` arrays on a latitude-longitude grid. There should be
                as many images as map components (:py:attr:`nc`). Users may
                provide :py:obj:`None` for any map component, which will set
                it to a uniform surface map. If :py:attr:`nc` is unity, a
                single item (string or ndarray) may be provided (instead of a
                one-element list).
            fix_amp: If True, this method will not change the amplitude of
                any of the spectral components. If False, the amplitude of
                each component will be proportional to the mean intensity
                of the corresponding image. Default is True.
            extent (tuple, optional): The lat-lon values corresponding to the
                edges of the image in degrees, ``(lat0, lat1, lon0, lon1)``.
                Default is ``(-180, 180, -90, 90)``.
            smoothing (float, optional): Gaussian smoothing strength.
                Increase this value to suppress ringing or explicitly set to
                zero to disable smoothing. Default is ``1/self.ydeg``.
            fac (float, optional): Factor by which to oversample the image
                when applying the SHT. Default is ``1.0``. Increase this
                number for higher fidelity (at the expense of increased
                computational time).
            eps (float, optional): Regularization strength for the spherical
                harmonic transform. Default is ``1e-12``.
            force_psd (bool, optional): Force the map to be positive
                semi-definite? Default is False.
            kwargs (optional): Any other kwargs passed directly to
                :py:meth:`minimize` (only if ``force_psd`` is True).

        """
        # Args checks
        assert self._ydeg > 0, "Can only load maps if ``ydeg`` > 0."
        msg = (
            "The map must be provided as a list of ``nc``"
            "file names or as a numerical array of length ``nc``."
        )
        assert not is_tensor(images), msg
        if self._nc > 1:
            assert hasattr(images, "__len__"), msg
            assert len(images) == self._nc, msg
        else:
            if type(images) is str or not hasattr(images, "__len__"):
                images = [images]
            elif type(images) is np.ndarray and images.ndim == 1:
                images = [images]

        # Load
        for n, map_n in enumerate(images):
            self._map.amp = 1.0
            if (map_n is None) or (
                type(map_n) is str and map_n.lower() == "none"
            ):
                self._map[1:, :] = 0.0
            else:
                self._map.load(map_n, **kwargs)
            if self._nc == 1:
                self[:, :] = self._map[:, :]
                if not fix_amp:
                    self._amp = self._math.reshape(
                        self.ops.set_vector(
                            self._math.reshape(self._amp, (self.nc,)),
                            0,
                            self._map.amp,
                        ),
                        (1, self.nc),
                    )
            else:
                self[:, :, n] = self._math.reshape(self._map[:, :], [-1, 1])
                if not fix_amp:
                    self._amp = self._math.reshape(
                        self.ops.set_vector(
                            self._math.reshape(self._amp, (self.nc,)),
                            n,
                            self._map.amp,
                        ),
                        (1, self.nc),
                    )

    def spot(self, *, component=0, **kwargs):
        r"""Add the expansion of a circular spot to the map.

        This function adds a spot whose functional form is a top
        hat in :math:`\Delta\theta`, the
        angular separation between the center of the spot and another
        point on the surface. The spot intensity is controlled by the
        parameter ``contrast``, defined as the fractional change in the
        intensity at the center of the spot.

        Args:
            component (int, optional): Indicates which spectral component
                to add the spot to. Default is ``0``.
            contrast (scalar or vector, optional): The contrast of the spot.
                This is equal to the fractional change in the intensity of the
                map at the *center* of the spot relative to the baseline intensity
                of an unspotted map. If the map has more than one
                wavelength bin, this must be a vector of length equal to the
                number of wavelength bins. Positive values of the contrast
                result in dark spots; negative values result in bright
                spots. Default is ``1.0``, corresponding to a spot with
                central intensity close to zero.
            radius (scalar, optional): The angular radius of the spot in
                units of :py:attr:`angle_unit`. Defaults to ``20.0`` degrees.
            lat (scalar, optional): The latitude of the spot in units of
                :py:attr:`angle_unit`. Defaults to ``0.0``.
            lon (scalar, optional): The longitude of the spot in units of
                :py:attr:`angle_unit`. Defaults to ``0.0``.

        .. note::

            Keep in mind that things are normalized in ``starry`` such that
            the disk-integrated *flux* (not the *intensity*!)
            of an unspotted body is unity. The default intensity of an
            unspotted map is ``1.0 / np.pi`` everywhere (this ensures the
            integral over the unit disk is unity).
            So when you instantiate a map and add a spot of contrast ``c``,
            you'll see that the intensity at the center is actually
            ``(1 - c) / np.pi``. This is expected behavior, since that's
            a factor of ``1 - c`` smaller than the baseline intensity.

        .. note::

            This function computes the spherical harmonic expansion of a
            circular spot with uniform contrast. At finite spherical
            harmonic degree, this will return an *approximation* that
            may be subject to ringing. Users can control the amount of
            ringing and the smoothness of the spot profile (see below).
            In general, however, at a given spherical harmonic degree
            ``ydeg``, there is always minimum spot radius that can be
            modeled well. For ``ydeg = 15``, for instance, that radius
            is about ``10`` degrees. Attempting to add a spot smaller
            than this will in general result in a large amount of ringing and
            a smaller contrast than desired.

        There are a few additional under-the-hood keywords
        that control the behavior of the spot expansion. These are

        Args:
            spot_pts (int, optional): The number of points in the expansion
                of the (1-dimensional) spot profile. Default is ``1000``.
            spot_eps (float, optional): Regularization parameter in the
                expansion. Default is ``1e-9``.
            spot_smoothing (float, optional): Standard deviation of the
                Gaussian smoothing applied to the spot to suppress
                ringing (unitless). Default is ``2.0 / self.ydeg``.
            spot_fac (float, optional): Parameter controlling the smoothness
                of the spot profile. Increasing this parameter increases
                the steepness of the profile (which approaches a top hat
                as ``spot_fac -> inf``). Decreasing it results in a smoother
                sigmoidal function. Default is ``300``. Changing this
                parameter is not recommended; change ``spot_smoothing``
                instead.

        .. note::

            These last four parameters are cached. That means that
            changing their value in a call to ``spot`` will result in
            all future calls to ``spot`` "remembering" those settings,
            unless you change them back!

        """
        self._map.spot(**kwargs)
        if self.nc == 1:
            self[:, :] = self._map[:, :]
        else:
            self[:, :, component] = self._map[:, :]

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
                self._inc, theta, self._veq, self._u, self._amp * self._y
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
                self._inc,
                theta,
                self._veq,
                self._u,
                self._amp * self._y,
                self._spectrum,
            )
        elif method == "convdot":
            flux = self.ops.get_flux_from_convdot(
                self._inc,
                theta,
                self._veq,
                self._u,
                self._amp * self._y,
                self._spectrum,
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
                    self._inc, theta, self._veq, self._u, self._amp * self._y
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
            self._inc, theta, self._veq, self._u, self._amp * self._y
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
        Importantly, if ``fix_spectrum`` is True, it is assumed that ``x``
        is a representation of the **amplitude-weighted** spatial map; i.e.,
        this method assumes it's the *spatial map* that carries the
        normalization (and not the *spectrum*).

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
                    self._inc,
                    theta,
                    self._veq,
                    self._u,
                    self._amp * self._y,
                    x,
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
                    self._inc,
                    theta,
                    self._veq,
                    self._u,
                    self._amp * self._y,
                    x,
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
                if self._nc == 1:
                    self._map[:, :] = self[:, :]
                    self._map.amp = self._amp[0, 0]
                else:
                    self._map[:, :] = self._math.reshape(self[:, :, k], (-1,))
                    self._map.amp = self._amp[0, k]
                img = get_val(self._map.render(projection="moll", res=res))
                moll[k] = img / np.nanmax(img)
                ortho += get_val(
                    self._map.render(
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
                get_val(self._math.transpose(self._amp) * self.spectrum),
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
        spatial_cho_cov=None,
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
        # Process defaults
        if flux_err is None:
            flux_err = 1e-6
        if spatial_mean is None:
            spatial_mean = 0.0
        if spatial_cov is None and spatial_cho_cov is None:
            spatial_cov = 1e-3 * np.ones(self.Ny)
            spatial_cov[0] = 1

        # Flux should be provided as a matrix of shape (nt, nw)
        # Internally, we unroll it into a vector
        flux = self._math.reshape(self._math.cast(flux), [self.nt * self.nw])

        # Flux error may be a scalar or a matrix
        # Internally, we unroll it into a vector
        flux_err = self._math.cast(flux_err)
        if flux_err.ndim == 0:
            flux_err = flux_err * self._math.ones([self.nt * self.nw])
        else:
            flux_err = self._math.reshape(flux_err, [self.nt * self.nw])

        # Spatial mean may be a scalar, a vector, or a matrix
        spatial_mean = self._math.cast(spatial_mean)
        if spatial_mean.ndim == 0:
            # The `solve` method accepts scalars, so we're good
            pass
        elif spatial_mean.ndim == 1:
            # Assume the user provided the mean for only the first
            # component. Copy it over to all components, then unroll
            # into a vector
            spatial_mean = self._math.reshape(
                self._math.tile(
                    self._math.reshape(spatial_mean, [self.Ny, 1]),
                    [1, self.nc],
                ),
                [self.Ny * self.nc],
            )
        else:
            # User specified the full (nc, Ny) matrix. Unroll it
            # into a vector.
            spatial_mean = self._math.reshape(
                spatial_mean, [self.nc * self.Ny]
            )

        # Spatial cov may be a scalar, a vector, or a matrix
        spatial_cov = self._math.cast(spatial_cov)
        if spatial_cov is not None:
            if spatial_cov.ndim == 0:
                # The `solve` method accepts scalars, so we're good
                pass
            elif spatial_cov.ndim == 1:
                # Assume the user provided the variance for only the first
                # component. Copy it over to all components.
                spatial_cov = self._math.reshape(
                    self._math.tile(
                        self._math.reshape(spatial_cov, [self.Ny, 1]),
                        [1, self.nc],
                    ),
                    [self.Ny * self.nc],
                )
            else:
                # User specified a full covariance matrix. This
                # is either a (Ny, Ny) or (nc * Ny, nc * Ny) matrix.
                # In the former case, we need to tile it.
                # TODO!!!
                # Remember that we need (nc, Ny), NOT (Ny, nc)!
                raise NotImplementedError(
                    "Dense covariance matrices not yet supported."
                )

        if fix_spectrum:

            # Get the design matrix conditioned on the current spectrum
            D = self.design_matrix(theta=theta, fix_spectrum=True)

            if (not normalized) or (baseline is not None):

                # The problem is exactly linear.

                if normalized:

                    # De-normalize the data w/ the given baseline
                    baseline = self._math.reshape(baseline, (self.nt, 1))
                    flux = flux * baseline
                    flux_err = flux_err * baseline

                # Solve the L2 problem
                mean, cho_cov = solve(
                    D,
                    flux,
                    C=flux_err ** 2,
                    cho_C=None,
                    mu=spatial_mean,
                    L=spatial_cov,
                    cho_L=spatial_cho_cov,
                    N=self.nc * self.Ny,
                    lazy=self.lazy,
                )

                # Set the current map to the MAP
                self[:, :, :] = self._math.transpose(
                    self._math.reshape(mean, (self.nc, self.Ny))
                )

                # Return the factorized posterior covariance
                # TODO: Reshape me?
                return cho_cov

            else:

                # TODO
                raise NotImplementedError("Not yet implemented.")

            return cho_C

        else:

            # TODO
            raise NotImplementedError("Not yet implemented.")
