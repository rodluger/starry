# -*- coding: utf-8 -*-
from . import config
from ._constants import *
from ._core import OpsDoppler, math
from ._core.utils import is_tensor, CompileLogMessage
from ._indices import integers, get_ylm_inds, get_ylmw_inds
from .compat import evaluator
from .maps import YlmBase, MapBase, Map
from .doppler_visualize import Visualize
import numpy as np


class DopplerMap:
    """
    The Doppler ``starry`` map class.

    """

    _clight = 299792458.0  # m/s

    def __init__(
        self,
        ydeg=0,
        udeg=0,
        nc=1,
        nt=10,
        wav1=642.5,  # Fe 6430
        wav2=643.5,
        nw=199,
        vsini_max=None,
        lazy=None,
        **kwargs
    ):
        # Check args
        ydeg = int(ydeg)
        assert ydeg >= 0, "Keyword `ydeg` must be positive."
        udeg = int(udeg)
        assert udeg >= 0, "Keyword `udeg` must be positive."
        assert nc is not None, "Please specify the number of map components."
        nc = int(nc)
        assert nc > 0, "Number of map components must be positive."
        assert nw is not None, "Please specify the number of wavelength bins."
        nw = int(nw)
        assert nw > 0, "Number of wavelength bins must be positive."
        # Enforce a minimum number of bins
        if nw < 99:
            nw = 99
        # Enforce an odd number of bins
        if nw % 2 == 0:
            nw += 1
        nt = int(nt)
        assert nt > 0, "Number of epochs must be positive."
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
        self._nw = nw

        # Wavelength grid (uniform in log)
        assert not is_tensor(
            wav1, wav2
        ), "Wavelength bounds must be numerical quantities."
        wavr = np.exp(0.5 * (np.log(wav1) + np.log(wav2)))
        log_lambda = np.linspace(np.log(wav1 / wavr), np.log(wav2 / wavr), nw)

        # This parameter determines the convolution kernel width
        self.velocity_unit = kwargs.pop("velocity_unit", units.m / units.s)
        if vsini_max is None:
            vsini_max = 1e5  # m/s = 100 km/s
        else:
            vsini_max *= self._velocity_factor

        # Compute the padded wavelength grid. We add bins corresponding to
        # the maximum kernel width to each end to prevent edge effects
        dlam = log_lambda[1] - log_lambda[0]
        betasini_max = vsini_max / self._clight
        hw = np.array(
            np.floor(
                np.abs(0.5 * np.log((1 + betasini_max) / (1 - betasini_max)))
                / dlam
            ),
            dtype="int32",
        )
        x = np.arange(0, hw + 1) * dlam
        pad_l = log_lambda[0] - hw * dlam + x[:-1]
        pad_r = log_lambda[-1] + x[1:]
        log_lambda_padded = np.concatenate([pad_l, log_lambda, pad_r])
        nwp = len(log_lambda_padded)

        # Store
        self.vsini_max = self._math.cast(vsini_max)
        self.log_lambda = self._math.cast(log_lambda)
        self._lambda = wavr * self._math.cast(np.exp(log_lambda))
        self._log_lambda_padded = self._math.cast(log_lambda_padded)
        self._lambda_padded = wavr * self._math.cast(np.exp(log_lambda_padded))
        self._nwp = nwp

        # Default spectrum (one centered absorption line, sigma = 1/2 Angstrom)
        spectrum = np.ones((nwp, nc))
        spectrum[:, 0] = 1 - 0.5 * np.exp(
            -0.5 * (wavr * np.exp(log_lambda_padded) - wavr) ** 2 / 0.05 ** 2
        )
        self._default_spectrum = spectrum

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
            log_lambda_padded,
            **kwargs
        )

        # Support map (for certain operations like `load`, `show`, etc.)
        # This map reflects all the properties of the DopplerMap except
        # the spherical harmonic coefficients `y`; these are set on an
        # as-needed basis.
        self._map = Map(ydeg=self.ydeg, udeg=self.udeg, lazy=self.lazy)

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

        # Basic properties
        self.inc = kwargs.pop("inc", 0.5 * np.pi / self._angle_factor)
        self.obl = kwargs.pop("obl", 0.0)
        self.veq = kwargs.pop("veq", 0.0)
        self.spectrum = kwargs.pop("spectrum", self._default_spectrum)

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
    def nwf(self):
        """Number of wavelength bins in the output (``flux``). *Read-only*"""
        return self._nw

    @property
    def nws(self):
        """Number of wavelength bins in the input (``spectrum``). *Read-only*"""
        return self._nwp

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
    def wavf(self):
        """
        The output wavelength grid. *Read-only*

        """
        return self._lambda

    @property
    def wavs(self):
        """
        The input wavelength grid. *Read-only*

        """
        return self._lambda_padded

    @property
    def spectrum(self):
        """
        The rest frame spectrum at wavelengths ``wav`` for each component.

        """
        return self._math.transpose(self._spectrum)

    @spectrum.setter
    def spectrum(self, value):
        self._spectrum = self.ops.enforce_shape(
            self._math.transpose(self._math.cast(value)),
            np.array([self._nc, self._nwp]),
        )

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
        The spectral-spatial map.

        This is equal to the outer product of the spherical harmonic
        decompositions and their corresponding spectral components. Dot
        the design matrix into this quantity to obtain the observed
        spectral timeseries (the ``flux``).
        """
        # Outer product with the map
        return self._math.dot(
            self._math.reshape(self._y, [self._Ny, self._nc]), self._spectrum
        )

    def design_matrix(self, theta):
        """Return the Doppler operator."""
        theta = (
            self.ops.enforce_shape(
                self._math.cast(theta), np.array([self._nt])
            )
            * self._angle_factor
        )
        D = self.ops.get_D(self._inc, theta, self._veq)
        return D

    def flux(self, theta):
        """Return the model for the full spectral timeseries."""
        theta = (
            self.ops.enforce_shape(
                self._math.cast(theta), np.array([self._nt])
            )
            * self._angle_factor
        )
        flux = self.ops.get_flux(
            self._inc, theta, self._veq, self.spectral_map
        )
        return flux

    def show(self, theta=None, res=150, file=None, **kwargs):
        """

        """
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

        # Get the observed spectrum at each phase
        flux = get_val(self.flux(theta / self._angle_factor))

        # Launch the web app
        viz = Visualize(
            get_val(self.wavs),
            get_val(self.wavf),
            moll,
            ortho,
            get_val(self.spectrum).T,
            theta,
            flux,
            get_val(self._inc),
        )
        if file is None:
            viz.launch()
        else:
            viz.save(file=file)
