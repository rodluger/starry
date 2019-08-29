# -*- coding: utf-8 -*-
from .. import config
from .maps import MapBase
from ..ops import to_tensor, autocompile, MapVector, vectorize
import numpy as np
from astropy import units, constants
from inspect import getmro
import theano.tensor as tt

try:
    # starry requires exoplanet >= v0.2.0
    from packaging import version
    import exoplanet

    if version.parse(exoplanet.__version__) < version.parse("0.2.0"):
        exoplanet = None
except ModuleNotFoundError:
    exoplanet = None


G_grav = constants.G.to(units.R_sun ** 3 / units.M_sun / units.day ** 2).value


__all__ = ["Primary", "Secondary", "System"]


class Body(object):
    def __init__(
        self,
        map,
        r=1.0,
        m=1.0,
        prot=1.0,
        t0=0.0,
        L=None,
        length_unit=units.Rsun,
        mass_unit=units.Msun,
        time_unit=units.day,
        angle_unit=units.degree,
        **kwargs
    ):
        # Surface map
        self.map = map

        # Units
        self.length_unit = length_unit
        self.mass_unit = mass_unit
        self.time_unit = time_unit
        self.angle_unit = angle_unit

        # Attributes
        self.r = r
        self.m = m
        self.prot = prot
        self.t0 = t0
        if L is not None:
            self.L = L

    @property
    def length_unit(self):
        """An ``astropy.units`` unit defining the length metric for this body."""
        return self._length_unit

    @length_unit.setter
    def length_unit(self, value):
        assert value.physical_type == "length"
        self._length_unit = value
        self._length_factor = value.in_units(units.Rsun)

    @property
    def mass_unit(self):
        """An ``astropy.units`` unit defining the mass metric for this body."""
        return self._mass_unit

    @mass_unit.setter
    def mass_unit(self, value):
        assert value.physical_type == "mass"
        self._mass_unit = value
        self._mass_factor = value.in_units(units.Msun)

    @property
    def time_unit(self):
        """An ``astropy.units`` unit defining the time metric for this body."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, value):
        assert value.physical_type == "time"
        self._time_unit = value
        self._time_factor = value.in_units(units.day)

    @property
    def angle_unit(self):
        """An ``astropy.units`` unit defining the angle metric for this body."""
        return self._angle_unit

    @angle_unit.setter
    def angle_unit(self, value):
        assert value.physical_type == "angle"
        self._angle_unit = value
        self._angle_factor = value.in_units(units.radian)

    @property
    def _angle_unit(self):
        return self._map._angle_unit

    @_angle_unit.setter
    def _angle_unit(self, value):
        self._map._angle_unit = value

    @property
    def _angle_factor(self):
        return self._map._angle_factor

    @_angle_factor.setter
    def _angle_factor(self, value):
        self._map._angle_factor = value

    @property
    def map(self):
        """The surface map for this body."""
        return self._map

    @map.setter
    def map(self, value):
        assert MapBase in getmro(
            type(value)
        ), "The `map` attribute must be a `starry` map instance."
        self._map = value

    @property
    def r(self):
        """The radius in units of :py:attr:`length_unit`."""
        return self._r / self._length_factor

    @r.setter
    def r(self, value):
        self._r = self.cast(value * self._length_factor)

    @property
    def m(self):
        """The mass in units of :py:attr:`mass_unit`."""
        return self._m / self._mass_factor

    @m.setter
    def m(self, value):
        self._m = self.cast(value * self._mass_factor)

    @property
    def prot(self):
        """The rotation period in units of :py:attr:`time_unit`."""
        return self._prot / self._time_factor

    @prot.setter
    def prot(self, value):
        self._prot = self.cast(value * self._time_factor)

    @property
    def t0(self):
        """A reference time in units of :py:attr:`time_unit`."""
        # TODO: Think about this one
        return self._t0 / self._time_factor

    @t0.setter
    def t0(self, value):
        self._t0 = self.cast(value * self._time_factor)

    @property
    def L(self):
        """The body luminosity in arbitrary units."""
        return self._map.L

    @L.setter
    def L(self, value):
        self._map.L = value

    def cast(self, *args, **kwargs):
        return self._map.cast(*args, **kwargs)


class Primary(Body):
    def __init__(self, map, **kwargs):
        # Initialize `Body`
        super(Primary, self).__init__(map, **kwargs)


class Secondary(Body):
    def __init__(self, map, **kwargs):
        # Initialize `Body`
        super(Secondary, self).__init__(map, **kwargs)

        # Attributes
        if kwargs.get("porb", None) is not None:
            self.porb = kwargs.get("porb", None)
        elif kwargs.get("a", None) is not None:
            self.a = kwargs.get("a", None)
        else:
            raise ValueError("Must provide a value for either `porb` or `a`.")
        self.ecc = kwargs.get("ecc", 0.0)
        self.w = kwargs.get("w", 0.5 * np.pi / self._angle_factor)
        self.Omega = kwargs.get("Omega", 0.0)
        self.inc = kwargs.get("inc", 0.5 * np.pi / self._angle_factor)

    @property
    def porb(self):
        """The orbital period in units of :py:attr:`time_unit`.
        
        .. note:: 
            Setting this value overrides the value of :py:attr:`a`.
        """
        if self._porb == 0.0:
            return None
        else:
            return self._porb / self._time_factor

    @porb.setter
    def porb(self, value):
        self._porb = self.cast(value * self._time_factor)
        self._a = 0.0

    @property
    def a(self):
        """The semi-major axis in units of :py:attr:`length_unit`.
        
        .. note:: 
            Setting this value overrides the value of :py:attr:`porb`.
        """
        if self._a == 0.0:
            return None
        else:
            return self._a / self._length_factor

    @a.setter
    def a(self, value):
        self._a = self.cast(value * self._length_factor)
        self._porb = 0.0

    @property
    def ecc(self):
        """The orbital eccentricity."""
        return self._ecc

    @ecc.setter
    def ecc(self, value):
        self._ecc = value

    @property
    def w(self):
        """The longitude of pericenter in units of :py:attr:`angle_unit`."""
        return self._w / self._angle_factor

    @w.setter
    def w(self, value):
        self._w = self.cast(value * self._angle_factor)

    @property
    def omega(self):
        """Alias for the longitude of pericenter :py:attr:`w`."""
        return self.w

    @omega.setter
    def omega(self, value):
        self.w = value

    @property
    def Omega(self):
        """The longitude of ascending node in units of :py:attr:`angle_unit`."""
        return self._Omega / self._angle_factor

    @Omega.setter
    def Omega(self, value):
        self._Omega = self.cast(value * self._angle_factor)

    @property
    def inc(self):
        """The orbital inclination in units of :py:attr:`angle_unit`."""
        # TODO: resolve tension with rotational inclination
        return self._inc / self._angle_factor

    @inc.setter
    def inc(self, value):
        self._inc = self.cast(value * self._angle_factor)


class System(object):
    def __init__(self, primary, *secondaries, time_unit=units.day):
        # Require exoplanet
        assert exoplanet is not None, "This class requires exoplanet >= 0.2.0."

        # Units
        self.time_unit = time_unit

        # Members
        assert (
            type(primary) is Primary
        ), "Argument `primary` must be an instance of `Primary`."
        self._primary = primary
        for sec in secondaries:
            assert (
                type(sec) is Secondary
            ), "Argument `*secondaries` must be a sequence of `Secondary` instances."
        self._secondaries = secondaries

    def cast(self, *args, **kwargs):
        return self._primary._map.cast(*args, **kwargs)

    @property
    def time_unit(self):
        """An ``astropy.units`` unit defining the time metric for the system."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, value):
        assert value.physical_type == "time"
        self._time_unit = value
        self._time_factor = value.in_units(units.day)

    @property
    def primary(self):
        """The primary (central) object in the Keplerian system."""
        return self._primary

    @property
    def secondaries(self):
        """A list of the secondary (orbiting) object(s) in the Keplerian system."""
        return self._secondaries

    def flux(self, t):
        """Compute the system flux at times ``t``."""
        # TODO: Make this work in reflected light
        # TODO: Add an `rv` function
        if config.lazy:
            t = tt.reshape(t, [-1])
            make_array = tt.as_tensor_variable
        else:
            t = np.atleast_1d(t)
            make_array = np.array
        return self._flux(
            make_array(t) * self._time_factor,
            self._primary._r,
            self._primary._m,
            self._primary._prot,
            self._primary._t0,
            self._primary._map.L,
            self._primary._map._inc,
            self._primary._map._obl,
            self._primary._map._y,
            self._primary._map._u,
            self._primary._map._f,
            make_array([sec._r for sec in self._secondaries]),
            make_array([sec._m for sec in self._secondaries]),
            make_array([sec._prot for sec in self._secondaries]),
            make_array([sec._t0 for sec in self._secondaries]),
            make_array([sec._porb for sec in self._secondaries]),
            make_array([sec._a for sec in self._secondaries]),
            make_array([sec._ecc for sec in self._secondaries]),
            make_array([sec._w for sec in self._secondaries]),
            make_array([sec._Omega for sec in self._secondaries]),
            make_array([sec._inc for sec in self._secondaries]),
            make_array([sec._map.L for sec in self._secondaries]),
            make_array([sec._map._inc for sec in self._secondaries]),
            make_array([sec._map._obl for sec in self._secondaries]),
            make_array([sec._map._y for sec in self._secondaries]),
            make_array([sec._map._u for sec in self._secondaries]),
            make_array([sec._map._f for sec in self._secondaries]),
        )

    @autocompile(
        "flux",
        tt.dvector(),  # t
        # -- primary --
        tt.dscalar(),  # r
        tt.dscalar(),  # m
        tt.dscalar(),  # prot
        tt.dscalar(),  # t0
        tt.dscalar(),  # L; TODO: this is a vector if `nw` > 1
        tt.dscalar(),  # inc
        tt.dscalar(),  # obl
        tt.dvector(),  # y; TODO: make this work for `nw` > 1
        tt.dvector(),  # u
        tt.dvector(),  # f
        # -- secondaries --
        tt.dvector(),  # r
        tt.dvector(),  # m
        tt.dvector(),  # prot
        tt.dvector(),  # t0
        tt.dvector(),  # porb
        tt.dvector(),  # a
        tt.dvector(),  # ecc
        tt.dvector(),  # w
        tt.dvector(),  # Omega
        tt.dvector(),  # iorb
        tt.dvector(),  # L; TODO: this is a matrix if `nw` > 1
        tt.dvector(),  # inc
        tt.dvector(),  # obl
        tt.dmatrix(),  # y; TODO: make this work for `nw` > 1
        tt.dmatrix(),  # u
        tt.dmatrix(),  # f
    )
    def _flux(
        self,
        t,
        pri_r,
        pri_m,
        pri_prot,
        pri_t0,
        pri_L,
        pri_inc,
        pri_obl,
        pri_y,
        pri_u,
        pri_f,
        sec_r,
        sec_m,
        sec_prot,
        sec_t0,
        sec_porb,
        sec_a,
        sec_ecc,
        sec_w,
        sec_Omega,
        sec_iorb,
        sec_L,
        sec_inc,
        sec_obl,
        sec_y,
        sec_u,
        sec_f,
    ):
        # Get all rotational phases
        theta_pri = (2 * np.pi) / pri_prot * (t - pri_t0)
        theta_sec = (
            (2 * np.pi)
            / tt.shape_padright(sec_prot)
            * (tt.shape_padleft(t) - tt.shape_padright(sec_t0))
        )

        # Compute all the phase curves
        phase_pri = pri_L * self._primary.map.ops.flux(
            theta_pri,
            tt.zeros_like(t),
            tt.zeros_like(t),
            tt.zeros_like(t),
            to_tensor(0.0),
            pri_inc,
            pri_obl,
            pri_y,
            pri_u,
            pri_f,
            no_compile=True,
        )
        phase_sec = tt.as_tensor_variable(
            [
                sec_L[i]
                * sec.map.ops.flux(
                    theta_sec[i],
                    tt.zeros_like(t),
                    tt.zeros_like(t),
                    tt.zeros_like(t),
                    to_tensor(0.0),
                    sec_inc[i],
                    sec_obl[i],
                    sec_y[i],
                    sec_u[i],
                    sec_f[i],
                    no_compile=True,
                )
                for i, sec in enumerate(self._secondaries)
            ]
        )

        # Compute any occultations
        occ_pri = tt.zeros_like(t)
        occ_sec = tt.zeros_like(phase_sec)

        # Compute the period if we were given a semi-major axis
        sec_porb = tt.switch(
            tt.eq(sec_porb, 0.0),
            (G_grav * (pri_m + sec_m) * sec_porb ** 2 / (4 * np.pi ** 2))
            ** (1.0 / 3),
            sec_porb,
        )

        # Compute the relative positions of all bodies
        orbit = exoplanet.orbits.KeplerianOrbit(
            period=sec_porb,
            t0=sec_t0,
            incl=sec_iorb,
            ecc=sec_ecc,
            omega=sec_w,
            Omega=sec_Omega,
            m_planet=sec_m,
            m_star=pri_m,
            r_star=pri_r,
            m_planet_units=units.Msun,
        )
        x, y, z = orbit.get_relative_position(t)

        # Compute transits across the primary
        for i, _ in enumerate(self._secondaries):
            xo = -x[:, i] / pri_r
            yo = -y[:, i] / pri_r
            zo = -z[:, i] / pri_r
            ro = sec_r[i] / pri_r
            b = tt.sqrt(xo ** 2 + yo ** 2)
            b_occ = tt.invert(
                tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
            )
            idx = tt.arange(b.shape[0])[b_occ]
            occ_pri = tt.set_subtensor(
                occ_pri[idx],
                occ_pri[idx]
                + pri_L
                * self._primary.map.ops.flux(
                    theta_pri[idx],
                    xo[idx],
                    yo[idx],
                    zo[idx],
                    ro,
                    pri_inc,
                    pri_obl,
                    pri_y,
                    pri_u,
                    pri_f,
                    no_compile=True,
                )
                - phase_pri[idx],
            )

        # Compute occultations by the primary
        for i, sec in enumerate(self._secondaries):
            xo = x[:, i] / sec_r[i]
            yo = y[:, i] / sec_r[i]
            zo = z[:, i] / sec_r[i]
            ro = pri_r / sec_r[i]
            b = tt.sqrt(xo ** 2 + yo ** 2)
            b_occ = tt.invert(
                tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
            )
            idx = tt.arange(b.shape[0])[b_occ]
            occ_sec = tt.set_subtensor(
                occ_sec[i, idx],
                occ_sec[i, idx]
                + sec_L[i]
                * sec.map.ops.flux(
                    theta_sec[i, idx],
                    xo[idx],
                    yo[idx],
                    zo[idx],
                    ro,
                    sec_inc[i],
                    sec_obl[i],
                    sec_y[i],
                    sec_u[i],
                    sec_f[i],
                    no_compile=True,
                )
                - phase_sec[i, idx],
            )

        # TODO: secondary-secondary occultations

        # Sum it all up and return
        flux_total = (
            phase_pri
            + occ_pri
            + tt.sum(phase_sec, axis=0)
            + tt.sum(occ_sec, axis=0)
        )
        return flux_total
