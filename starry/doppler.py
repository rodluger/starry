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
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import csr_matrix
from warnings import warn


class DopplerMap:
    """
    The Doppler ``starry`` map class.

    """

    _clight = 299792458.0  # m/s
    _default_vsini_max = 1e5  # m/s
    _default_wav = np.linspace(642.5, 643.5, 200)  # FeI 6430

    def _default_spectrum(self):
        spectrum = self._math.ones((self._nc, self._nw0))
        spectrum[0] = 1 - 0.5 * self._math.exp(
            -0.5 * (self._wav0 - self._wavr) ** 2 / 0.05 ** 2
        )
        return spectrum

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
        assert ydeg >= 1, "Keyword `ydeg` must be >= 1."
        udeg = int(udeg)
        assert udeg >= 0, "Keyword `udeg` must be positive."
        assert nc is not None, "Please specify the number of map components."
        nc = int(nc)
        assert nc > 0, "Number of map components must be positive."
        nt = int(nt)
        assert nt > 0, "Number of epochs must be positive."
        assert (
            interp_order >= 1 and interp_order <= 5
        ), "Keyword `interp_order` must be in the range [1, 5]."
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

        # Compute the padded internal wavelength grid (wav_int_padded).
        # We add bins corresponding to the maximum kernel width to each
        # end of wav_int to prevent edge effects
        dlam = log_wav[1] - log_wav[0]
        betasini_max = vsini_max / self._clight
        hw = np.array(
            np.floor(
                np.abs(0.5 * np.log((1 + betasini_max) / (1 - betasini_max)))
                / dlam
            ),
            dtype="int32",
        )
        x = np.arange(0, hw + 1) * dlam
        pad_l = log_wav[0] - hw * dlam + x[:-1]
        pad_r = log_wav[-1] + x[1:]
        log_wav_padded = np.concatenate([pad_l, log_wav, pad_r])
        wav_int_padded = wavr * np.exp(log_wav_padded)
        nwp = len(log_wav_padded)
        self._log_wav_padded = self._math.cast(log_wav_padded)
        self._wav_int_padded = self._math.cast(wav_int_padded)
        self._nw_int_padded = nwp

        # Compute the user-facing rest spectrum wavelength grid (wav0)
        assert not is_tensor(
            wav0
        ), "Wavelength grids must be numerical quantities."
        if wav0 is None:
            # The default grid is the data wavelength grid with
            # a bit of padding on either side
            delta_wav = np.median(np.diff(np.sort(wav)))
            pad_l = np.arange(wav1, wav_int_padded[0] - delta_wav, -delta_wav)
            pad_l = pad_l[::-1][:-1]
            pad_r = np.arange(wav2, wav_int_padded[-1] + delta_wav, delta_wav)
            pad_r = pad_r[1:]
            wav0 = np.concatenate([pad_l, wav, pad_r])
        wav0 = np.array(wav0)
        nw0 = len(wav0)
        self._wav0 = self._math.cast(wav0)
        self._nw0 = nw0
        if (wav_int_padded[0] < np.min(wav0)) or (
            wav_int_padded[-1] > np.max(wav0)
        ):
            warn(
                "Rest frame wavelength grid `wav0` is not sufficiently padded. "
                "Edge effects may occur. See the documentation for mode details."
            )

        # Interpolation between internal grid and user grid
        self._interp = interpolate
        self._interp_order = interp_order
        self._interp_tol = interp_tol
        if self._interp:

            # Compute the flux interpolation operator (wav <-- wav_int)
            # `S` interpolates the flux back onto the user-facing `wav` grid
            # `SBlock` interpolates the design matrix onto the `wav grid`
            S = self._get_spline_operator(wav_int, wav)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._S = self._math.sparse_cast(S.T)
            self._SBlock = self._math.sparse_cast(
                sparse_block_diag([S for n in range(nt)])
            )

            # Compute the spec interpolation operator (wav0 <-- wav_int_padded)
            # `S0` interpolates the user-provided spectrum onto the internal grid
            # `S0Inv` performs the inverse operation
            S = self._get_spline_operator(wav_int_padded, wav0)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._S0 = self._math.sparse_cast(S.T)
            S = self._get_spline_operator(wav0, wav_int_padded)
            S[np.abs(S) < interp_tol] = 0
            S = csr_matrix(S)
            self._S0Inv = self._math.sparse_cast(S.T)

        else:

            # No interpolation. User-facing grids *are* the internal grids
            # User will handle interpolation on their own
            self._wav = self._wav_int
            self._wav0 = self._wav_int_padded
            self._nw0 = self._nw_int_padded

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
            log_wav_padded,
            **kwargs
        )

        # Support map (for certain operations like `load`, `show`, etc.)
        # This map reflects all the properties of the DopplerMap except
        # the spherical harmonic coefficients `y`; these are set on an
        # as-needed basis.
        _quiet = config.quiet
        config.quiet = True
        self._map = Map(ydeg=self.ydeg, udeg=self.udeg, lazy=self.lazy)
        config.quiet = _quiet

        # Initialize
        self.reset(**kwargs)

    def reset(self, **kwargs):

        # Units
        self.angle_unit = kwargs.pop("angle_unit", units.degree)
        self.velocity_unit = kwargs.pop("velocity_unit", units.m / units.s)

        # Map properties
        if self._nc == 1:
            y = np.zeros(self._Ny)
            y[0] = 1.0
        else:
            y = np.zeros((self._Ny, self._nc))
            y[0, :] = 1.0
        self._y = self._math.cast(y)
        u = np.zeros(self._Nu)
        u[0] = -1.0
        self._u = self._math.cast(u)
        self._map._u = self._u

        # Reset the spectrum
        self.spectrum = kwargs.pop("spectrum", self._default_spectrum())

        # Basic properties
        self.inc = kwargs.pop("inc", 0.5 * np.pi / self._angle_factor)
        self.obl = kwargs.pop("obl", 0.0)
        self.veq = kwargs.pop("veq", 0.0)

    @property
    def lazy(self):
        """Map evaluation mode: lazy or greedy?"""
        return self._lazy

    @property
    def nc(self):
        """Number of map components. *Read-only*"""
        return self._nc

    @property
    def nt(self):
        """Number of spectral epochs. *Read-only*"""
        return self._nt

    @property
    def nw(self):
        """Length of the user-facing flux wavelength grid `wav`. *Read-only*"""
        return self._nw

    @property
    def nw0(self):
        """Length of the user-facing rest frame spectrum wavelength grid `wav0`. *Read-only*"""
        return self._nw0

    @property
    def nw_internal(self):
        """Length of the *internal* flux wavelength grid `wav`. *Read-only*"""
        return self._nw_int

    @property
    def nw0_internal(self):
        """Length of the *internal* rest frame spectrum wavelength grid `wav0`. *Read-only*"""
        return self._nw_int_padded

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
        """Limb darkening degree. *Read-only*"""
        return self._udeg

    @property
    def Nu(self):
        r"""Number of limb darkening coefficients, including :math:`u_0`. *Read-only*

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
        """The inclination of the rotation axis in units of :py:attr:`angle_unit`."""
        return self._inc / self._angle_factor

    @inc.setter
    def inc(self, value):
        self._inc = self._math.cast(value) * self._angle_factor
        self._map._inc = self._inc

    @property
    def obl(self):
        """The obliquity of the rotation axis in units of :py:attr:`angle_unit`."""
        return self._obl / self._angle_factor

    @obl.setter
    def obl(self, value):
        self._obl = self._math.cast(value) * self._angle_factor

    @property
    def veq(self):
        """The equatorial velocity of the body in units of :py:attr:`velocity_unit`.
        """
        return self._veq / self._velocity_factor

    @veq.setter
    def veq(self, value):
        self._veq = self._math.cast(value) * self._velocity_factor

    @property
    def vsini(self):
        """
        The projected equatorial radial velocity in units of ``velocity_unit``.
        *Read-only*

        """
        return self._veq * self._math.sin(self._inc) / self._velocity_factor

    def load(self, map, **kwargs):
        """
        Load the spatial map(s).

        """
        # Args checks
        assert self._ydeg > 0, "Can only load maps if ``ydeg`` > 0."
        msg = (
            "The map must be provided as a list of ``nc``"
            "file names or as a numerical array of length ``nc``."
        )
        assert not is_tensor(map), msg
        if self._nc > 1:
            assert hasattr(map, "__len__"), msg
            assert len(map) == self._nc, msg
        else:
            if type(map) is str or not hasattr(map, "__len__"):
                map = [map]
            elif type(map) is np.ndarray and map.ndim == 1:
                map = [map]

        # Load
        for n, map_n in enumerate(map):
            if (map_n is None) or (
                type(map_n) is str and map_n.lower() == "none"
            ):
                self._map[1:, :] = 0.0
            else:
                self._map.load(map_n, **kwargs)
            if self._nc == 1:
                self[1:, :] = self._map[1:, :]
            else:
                self[1:, :, n] = self._math.reshape(self._map[1:, :], [-1, 1])

    @property
    def wav(self):
        """
        The output wavelength grid. *Read-only*

        This is the wavelength grid on which quantities like the ``flux``
        and ``design_matrix`` are defined.

        """
        return self._wav

    @property
    def wav0(self):
        """
        The rest-frame wavelength grid. *Read-only*

        This is the wavelength grid on which the ``spectrum`` is defined.

        """
        return self._wav0

    @property
    def spectrum(self):
        """
        The rest frame spectrum for each component.

        This quantity is defined on the wavelength grid ``wav0``.

        Shape must be ``(nc, len(wav0))``. If ``nc = 1``, a one-dimensional
        array of length ``len(wav0)`` is also accepted.

        """
        # Interpolate to the `wav0` grid
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

        # Interpolate from `wav0` grid to internal, padded grid
        if self._interp:
            self._spectrum = self._math.sparse_dot(spectrum, self._S0Inv)
        else:
            self._spectrum = spectrum

    @property
    def y(self):
        """The spherical harmonic coefficient vector. *Read-only*

        To set this vector, index the map directly using two indices:
        ``map[l, m] = ...`` where ``l`` is the spherical harmonic degree and
        ``m`` is the spherical harmonic order. These may be integers or
        arrays of integers. Slice notation may also be used.

        If ``nc > 1``, index the map directly using three indices instead:
        ``map[l, m, c] = ...`` where ``c`` is the index of the map component.
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
        elif isinstance(idx, tuple) and len(idx) == 2 and self.nc == 1:
            # User is accessing a Ylm index
            inds = get_ylm_inds(self.ydeg, idx[0], idx[1])
            return self._y[inds]
        elif isinstance(idx, tuple) and len(idx) == 3 and self.nc > 1:
            # User is accessing a Ylmc index
            inds = get_ylmw_inds(self.ydeg, self.nc, idx[0], idx[1], idx[2])
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
            if self.lazy:
                self._u = self.ops.set_map_vector(self._u, inds, val)
            else:
                self._u[inds] = val
            self._map._u = self._u
        elif isinstance(idx, tuple) and len(idx) == 2 and self.nc == 1:
            # User is accessing a Ylm index
            inds = get_ylm_inds(self.ydeg, idx[0], idx[1])
            if 0 in inds:
                if np.array_equal(np.sort(inds), np.arange(self.Ny)):
                    # The user is setting *all* coefficients, so we allow
                    # them to "set" the Y_{0,0} coefficient...
                    if self.lazy:
                        self._y = self.ops.set_map_vector(self._y, inds, val)
                    else:
                        self._y[inds] = val
                    # ... except we scale the amplitude of the map and
                    # force Y_{0,0} to be unity.
                    self.amp = self._y[0]
                    self._y /= self._y[0]
                else:
                    raise ValueError("The Y_{0,0} coefficient cannot be set.")
            else:
                if self.lazy:
                    self._y = self.ops.set_map_vector(self._y, inds, val)
                else:
                    self._y[inds] = val
        elif isinstance(idx, tuple) and len(idx) == 3 and self.nc > 1:
            # User is accessing a Ylmc index
            inds = get_ylmw_inds(self.ydeg, self.nc, idx[0], idx[1], idx[2])
            if 0 in inds[0]:
                raise ValueError("The Y_{0,0} coefficient cannot be set.")
            else:
                if self.lazy:
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

    @property
    def spectral_map(self):
        """
        The spectral-spatial map vector.

        This is equal to the (unrolled) outer product of the spherical harmonic
        decompositions and their corresponding spectral components. Dot
        the design matrix into this quantity to obtain the observed
        spectral timeseries (the ``flux``).
        """
        # Outer product with the map
        return self._math.reshape(
            self._math.dot(
                self._math.reshape(self._y, [self._Ny, self._nc]),
                self._spectrum,
            ),
            (-1,),
        )

    def spot(self, *, component=0, **kwargs):
        """

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
        """Return the Doppler imaging design matrix."""
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
            # TODO
            raise NotImplementedError("Not yet implemented.")

        else:

            # Full matrix (sparse)
            D = self.ops.get_D(self._inc, theta, self._veq, self._u)

        # Interpolate to the output grid
        if self._interp:
            D = self._math.sparse_dot(self._SBlock, D)

        return D

    def flux(self, theta=None, mode="convdot"):
        """
        Return the model for the full spectral timeseries.

        """
        theta = self._get_default_theta(theta)
        if mode == "convdot":
            flux = self.ops.get_flux_from_convdot(
                self._inc, theta, self._veq, self._u, self._y, self._spectrum
            )
        elif mode == "conv":
            flux = self.ops.get_flux_from_conv(
                self._inc, theta, self._veq, self._u, self.spectral_map
            )
        elif mode == "design":
            flux = self.ops.get_flux_from_design(
                self._inc, theta, self._veq, self._u, self.spectral_map
            )
        else:
            raise ValueError("Keyword `mode` must be one of `conv`, `design`.")

        # Interpolate to the output grid
        if self._interp:
            flux = self._math.sparse_dot(flux, self._S)

        return flux

    def dot(self, matrix, theta=None):
        """Dot the Doppler design matrix into a given matrix or vector."""
        theta = self._get_default_theta(theta)
        product = self.ops.dot_design_matrix_into(
            self._inc, theta, self._veq, self._u, self._math.cast(matrix)
        )

        # Interpolate to the output grid
        if self._interp:
            product = self._math.sparse_dot(self._SBlock, product)

        return product

    def show(self, theta=None, res=150, file=None, **kwargs):
        """

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
                    self._map._y = self._math.reshape(self[:, :], (-1,))
                else:
                    self._map._y = self._math.reshape(self[:, :, k], (-1,))
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
            # We'll normalize it to the median >90th percentile flux level
            veq = self.veq
            self.veq = 0.0
            flux0 = get_val(self.flux(theta / self._angle_factor))
            norm = np.nanmedian(
                np.sort(flux0, axis=-1)[:, int(0.9 * flux0.shape[-1]) :],
                axis=-1,
            )
            flux0 /= norm.reshape(-1, 1)
            self.veq = veq

            # Get the observed spectrum at each phase
            # We'll normalize it to the median >90th percentile flux level
            flux = get_val(self.flux(theta / self._angle_factor))
            norm = np.nanmedian(
                np.sort(flux, axis=-1)[:, int(0.9 * flux.shape[-1]) :], axis=-1
            )
            flux /= norm.reshape(-1, 1)

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
