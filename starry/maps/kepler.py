# -*- coding: utf-8 -*-
from .maps import MapBase
from ..ops import to_tensor
import numpy as np
from astropy import units
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
        self._update_orbit()

    @property
    def m(self):
        """The mass in units of :py:attr:`mass_unit`."""
        return self._m / self._mass_factor

    @m.setter
    def m(self, value):
        self._m = self.cast(value * self._mass_factor)
        self._update_orbit()

    @property
    def prot(self):
        """The rotation period in units of :py:attr:`time_unit`."""
        return self._prot / self._time_factor

    @prot.setter
    def prot(self, value):
        self._prot = self.cast(value * self._time_factor)
        self._update_orbit()

    @property
    def t0(self):
        """A reference time in units of :py:attr:`time_unit`."""
        # TODO: Think about this one
        return self._t0 / self._time_factor

    @t0.setter
    def t0(self, value):
        self._t0 = self.cast(value * self._time_factor)
        self._update_orbit()

    @property
    def L(self):
        """The body luminosity in arbitrary units."""
        return self._map.L

    @L.setter
    def L(self, value):
        self._map.L = value

    @property
    def lazy(self):
        return self._map.lazy

    def cast(self, *args, **kwargs):
        return self._map.cast(*args, **kwargs)

    def _update_orbit(self):
        pass


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
        if self._porb is None:
            return None
        else:
            return self._porb / self._time_factor

    @porb.setter
    def porb(self, value):
        self._porb = self.cast(value * self._time_factor)
        self._a = None
        self._update_orbit()

    @property
    def a(self):
        """The semi-major axis in units of :py:attr:`length_unit`.
        
        .. note:: 
            Setting this value overrides the value of :py:attr:`porb`.
        """
        if self._a is None:
            return None
        else:
            return self._a / self._length_factor

    @a.setter
    def a(self, value):
        self._a = self.cast(value * self._length_factor)
        self._porb = None
        self._update_orbit()

    @property
    def ecc(self):
        """The orbital eccentricity."""
        return self._ecc

    @ecc.setter
    def ecc(self, value):
        self._ecc = value
        self._update_orbit()

    @property
    def w(self):
        """The longitude of pericenter in units of :py:attr:`angle_unit`."""
        return self._w / self._angle_factor

    @w.setter
    def w(self, value):
        self._w = self.cast(value * self._angle_factor)
        self._update_orbit()

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
        self._update_orbit()

    @property
    def inc(self):
        """The orbital inclination in units of :py:attr:`angle_unit`."""
        # TODO: resolve tension with rotational inclination
        return self._inc / self._angle_factor

    @inc.setter
    def inc(self, value):
        self._inc = self.cast(value * self._angle_factor)
        self._update_orbit()


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
            assert (
                sec.lazy == self._primary.lazy
            ), "Mixing `lazy` and non-`lazy` bodies in a `System` class is not supported."
        self._secondaries = secondaries

        # Instantiate an orbit instance
        self._update_orbit()

        # Set up callbacks to update the orbit
        # TODO: ONLY if we're not in a pymc3 model context!!!
        self._primary._update_orbit = self._update_orbit
        for sec in self.secondaries:
            sec._update_orbit = self._update_orbit

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

    def _update_orbit(self):
        period = [sec._porb for sec in self._secondaries]
        if None in period:
            period = None
        a = [sec._a for sec in self._secondaries]
        if None in a:
            a = None
        if period is None and a is None:
            raise ValueError(
                "Please provide *either* periods or semi-major axes for all "
                "secondary instances. Mixing values is not supported."
            )
        self._orbit = exoplanet.orbits.KeplerianOrbit(
            period=period,
            a=a,
            t0=[sec._t0 for sec in self._secondaries],
            incl=[sec._inc for sec in self._secondaries],
            ecc=[sec._ecc for sec in self._secondaries],
            omega=[sec._w for sec in self._secondaries],
            Omega=[sec._Omega for sec in self._secondaries],
            m_planet=[sec._m for sec in self._secondaries],
            m_star=self._primary._m,
            r_star=self._primary._r,
            m_planet_units=units.Msun,
        )

    # TODO: autocompile!
    # TODO: Make this work when lazy is False
    def flux(self, t):
        """Compute the system flux at time `t` in units of :py:attr:`time_unit`."""
        # Get all rotational phases
        theta_pri = (
            (2 * np.pi)
            / self._primary._angle_factor
            / self._primary._prot
            * (self._primary.cast(t) * self._time_factor - self._primary._t0)
        )
        theta_sec = tt.as_tensor_variable(
            [
                (2 * np.pi)
                / sec._angle_factor
                / sec._prot
                * (sec.cast(t) * self._time_factor - sec._t0)
                for sec in self._secondaries
            ]
        )

        # Compute all the phase curves
        phase_pri = self._primary.map.flux(theta=theta_pri)
        phase_sec = tt.as_tensor_variable(
            [
                sec.map.flux(theta=theta_sec[i])
                for i, sec in enumerate(self._secondaries)
            ]
        )

        # Get the positions of all the bodies
        x, y, z = self._orbit.get_relative_position(
            to_tensor(t) * self._time_factor
        )

        # Compute transits across the primary
        occ_pri = tt.zeros_like(phase_pri)
        for i, sec in enumerate(self._secondaries):
            xo = -x[:, i] / self._primary._r
            yo = -y[:, i] / self._primary._r
            zo = -z[:, i] / self._primary._r
            ro = sec._r / self._primary._r
            b = tt.sqrt(xo ** 2 + yo ** 2)
            b_occ = tt.invert(
                tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
            )
            idx = tt.arange(b.shape[0])[b_occ]
            occ_pri = tt.set_subtensor(
                occ_pri[idx],
                occ_pri[idx]
                + self._primary.map.flux(
                    theta=theta_pri[idx],
                    xo=xo[idx],
                    yo=yo[idx],
                    zo=zo[idx],
                    ro=ro,
                )
                - phase_pri[idx],
            )

        # Compute occultations by the primary
        occ_sec = tt.zeros_like(phase_sec)

        for i, sec in enumerate(self._secondaries):
            xo = x[:, i] / sec._r
            yo = y[:, i] / sec._r
            zo = z[:, i] / sec._r
            ro = self._primary._r / sec._r
            b = tt.sqrt(xo ** 2 + yo ** 2)
            b_occ = tt.invert(
                tt.ge(b, 1.0 + ro) | tt.le(zo, 0.0) | tt.eq(ro, 0.0)
            )
            idx = tt.arange(b.shape[0])[b_occ]
            occ_sec = tt.set_subtensor(
                occ_sec[i, idx],
                occ_sec[i, idx]
                + sec.map.flux(
                    theta=theta_sec[i, idx],
                    xo=xo[idx],
                    yo=yo[idx],
                    zo=zo[idx],
                    ro=ro,
                )
                - phase_sec[i, idx],
            )

        # Sum it all up and return
        flux_total = (
            phase_pri
            + occ_pri
            + tt.sum(phase_sec, axis=0)
            + tt.sum(occ_sec, axis=0)
        )
        return flux_total
