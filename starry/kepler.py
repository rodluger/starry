# -*- coding: utf-8 -*-
"""
TODO:
    - Radial velocity support
    - Exposure time integration
    - Light travel time delay
    - Check kwargs
"""
from . import config
from .maps import MapBase, RVBase, ReflectedBase
from .ops import (
    OpsSystem,
    OpsRVSystem,
    G_grav,
    reshape,
    make_array_or_tensor,
    math,
)
import numpy as np
from astropy import units
from inspect import getmro


__all__ = ["Primary", "Secondary", "System"]


class Body(object):
    """A generic body. Must be subclassed."""

    def __init__(
        self,
        map,
        r=1.0,
        m=1.0,
        prot=1.0,
        t0=0.0,
        theta0=0.0,
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
        self.theta0 = theta0

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
        return self._t0 / self._time_factor

    @t0.setter
    def t0(self, value):
        self._t0 = self.cast(value * self._time_factor)

    @property
    def theta0(self):
        """The map rotational phase at time :py:attr:`t0`."""
        return self._theta0 / self._angle_factor

    @theta0.setter
    def theta0(self, value):
        self._theta0 = self.cast(value * self._angle_factor)

    def cast(self, *args, **kwargs):
        return self._map.cast(*args, **kwargs)


class Primary(Body):
    """A primary (central) body.
        
        Args:
            map: The surface map of this body. This should be an instance
                returned by :py:func:`starry.Map`.
            r (scalar, optional): The radius of the body in units of 
                :py:attr:`length_unit`. Defaults to 1.0.
            m (scalar, optional): The mass of the body in units of 
                :py:attr:`mass_unit`. Defaults to 1.0.
            prot (scalar, optional): The rotation period of the body in units of 
                :py:attr:`time_unit`. Defaults to 1.0.
            t0 (scalar, optional): A reference time in units of
                :py:attr:`time_unit`. Defaults to 0.0.
            theta0 (scalar, optional): The rotational phase of the map at time
                :py:attr:`t0` in units of :py:attr:`angle_unit`. Defaults to 0.0.
            length_unit (optional): An ``astropy.units`` unit defining the 
                distance metric for this object. Defaults to 
                :py:attr:`astropy.units.Rsun.`
            mass_unit (optional): An ``astropy.units`` unit defining the 
                mass metric for this object. Defaults to 
                :py:attr:`astropy.units.Msun.`
            time_unit (optional): An ``astropy.units`` unit defining the 
                time metric for this object. Defaults to 
                :py:attr:`astropy.units.day.`
            angle_unit (optional): An ``astropy.units`` unit defining the 
                angular metric for this object. Defaults to 
                :py:attr:`astropy.units.degree.`
    """

    def __init__(self, map, **kwargs):
        # Initialize `Body`
        super(Primary, self).__init__(map, **kwargs)


class Secondary(Body):
    """A secondary (orbiting) body.

        Args:
            map: The surface map of this body. This should be an instance
                returned by :py:func:`starry.Map`.
            r (scalar, optional): The radius of the body in units of 
                :py:attr:`length_unit`. Defaults to 1.0.
            m (scalar, optional): The mass of the body in units of 
                :py:attr:`mass_unit`. Defaults to 1.0.
            a (scalar, optional): The semi-major axis of the body in units of 
                :py:attr:`time_unit`. Defaults to 1.0. If :py:attr:`porb` is
                also provided, this value is ignored.
            porb (scalar, optional): The orbital period of the body in units of 
                :py:attr:`time_unit`. Defaults to 1.0. Setting this value 
                overrides :py:attr:`a`.
            prot (scalar, optional): The rotation period of the body in units of 
                :py:attr:`time_unit`. Defaults to 1.0.
            t0 (scalar, optional): A reference time in units of
                :py:attr:`time_unit`. This is taken to be the time of a reference
                transit. Defaults to 0.0.
            ecc (scalar, optional): The orbital eccentricity of the body.
                Defaults to 0.
            w, omega (scalar, optional): The argument of pericenter of the body
                in units of :py:attr:`angle_unit`. Defaults to 90 degrees.
            Omega (scalar, optional): The longitude of ascending node of the 
                body in units of :py:attr:`angle_unit`. Defaults to 0 degrees.
            inc (scalar, optional): The orbital inclination of the body in 
                units of :py:attr:`angle_unit`. Defaults to 90 degrees.
            theta0 (scalar, optional): The rotational phase of the map at time
                :py:attr:`t0` in units of :py:attr:`angle_unit`. Defaults to 
                0.0.
            length_unit (optional): An ``astropy.units`` unit defining the 
                distance metric for this object. Defaults to 
                :py:attr:`astropy.units.Rsun.`
            mass_unit (optional): An ``astropy.units`` unit defining the 
                mass metric for this object. Defaults to 
                :py:attr:`astropy.units.Msun.`
            time_unit (optional): An ``astropy.units`` unit defining the 
                time metric for this object. Defaults to 
                :py:attr:`astropy.units.day.`
            angle_unit (optional): An ``astropy.units`` unit defining the 
                angular metric for this object. Defaults to 
                :py:attr:`astropy.units.degree.`
    """

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
        self.w = kwargs.get(
            "w", kwargs.get("omega", 0.5 * np.pi / self._angle_factor)
        )
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
        return self._inc / self._angle_factor

    @inc.setter
    def inc(self, value):
        self._inc = self.cast(value * self._angle_factor)

    '''
    TODO: Is this property actually useful?
    @property
    def axis(self):
        """The axis perpendicular to the orbital plane. *Read-only.*

        This value is computed from :py:attr:`inc` and :py:attr:`Omega`.
        """
        cosO = math.cos(self._Omega)
        sinO = math.sin(self._Omega)
        cosI = math.cos(self._inc)
        sinI = math.sin(self._inc)
        return make_array_or_tensor([-sinO * sinI, cosO * sinI, cosI])
    '''


class System(object):
    """A system of bodies in Keplerian orbits about a central primary body.
    
    Args:
        primary (:py:class:`Primary`): The central body.
        secondaries (:py:class:`Secondary`): One or more secondary bodies
            in orbit about the primary. 
        time_unit (optional): An ``astropy.units`` unit defining the 
            time metric for this object. Defaults to 
            :py:attr:`astropy.units.day.`
        light_delay (bool, optional): Account for the light travel time 
            delay to the barycenter of the system? Default is False.
        quiet (bool, optional): Suppress information messages? 
            Defaults to False.
    """

    def __init__(
        self,
        primary,
        *secondaries,
        time_unit=units.day,
        light_delay=False,
        quiet=False
    ):
        # Units
        self.time_unit = time_unit

        # Primary body
        assert (
            type(primary) is Primary
        ), "Argument `primary` must be an instance of `Primary`."
        assert ReflectedBase not in getmro(
            type(primary._map)
        ), "Reflected light map not allowed for the primary body."
        self._primary = primary
        self._rv = RVBase in getmro(type(primary._map))

        # Secondary bodies
        assert len(secondaries) > 0, "There must be at least one secondary."
        for sec in secondaries:
            assert type(sec) is Secondary, (
                "Argument `*secondaries` must be a sequence of "
                "`Secondary` instances."
            )
            assert (
                sec._map.nw == self._primary._map.nw
            ), "All bodies must have the same number of wavelength bins `nw`."
            assert (RVBase in getmro(type(sec._map))) == self._rv, (
                "Radial velocity must be enabled "
                "for either all or none of the bodies."
            )

        reflected = [
            ReflectedBase in getmro(type(sec._map)) for sec in secondaries
        ]
        if np.all(reflected):
            self._reflected = True
        elif np.any(reflected):
            raise ValueError(
                "Reflected light must be enabled "
                "for either all or none of the secondaries."
            )
        else:
            self._reflected = False
        self._secondaries = secondaries

        # Theano ops class
        if self._rv:
            self.ops = OpsRVSystem(
                self._primary,
                self._secondaries,
                quiet=quiet,
                light_delay=light_delay,
            )
        else:
            self.ops = OpsSystem(
                self._primary,
                self._secondaries,
                reflected=self._reflected,
                quiet=quiet,
                light_delay=light_delay,
            )

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

    def X(self, t):
        """Compute the system flux design matrix at times ``t``.
        
        Args:
            t (scalar or vector): An array of times at which to evaluate
                the design matrix in units of :py:attr:`time_unit`.
        """
        return self.ops.X(
            reshape(make_array_or_tensor(t), [-1]) * self._time_factor,
            self._primary._r,
            self._primary._m,
            self._primary._prot,
            self._primary._t0,
            self._primary._theta0,
            self._primary._map._L,
            self._primary._map._inc,
            self._primary._map._obl,
            self._primary._map._u,
            self._primary._map._f,
            make_array_or_tensor([sec._r for sec in self._secondaries]),
            make_array_or_tensor([sec._m for sec in self._secondaries]),
            make_array_or_tensor([sec._prot for sec in self._secondaries]),
            make_array_or_tensor([sec._t0 for sec in self._secondaries]),
            make_array_or_tensor([sec._theta0 for sec in self._secondaries]),
            self._get_periods(),
            make_array_or_tensor([sec._ecc for sec in self._secondaries]),
            make_array_or_tensor([sec._w for sec in self._secondaries]),
            make_array_or_tensor([sec._Omega for sec in self._secondaries]),
            make_array_or_tensor([sec._inc for sec in self._secondaries]),
            make_array_or_tensor([sec._map._L for sec in self._secondaries]),
            make_array_or_tensor([sec._map._inc for sec in self._secondaries]),
            make_array_or_tensor([sec._map._obl for sec in self._secondaries]),
            make_array_or_tensor([sec._map._u for sec in self._secondaries]),
            make_array_or_tensor([sec._map._f for sec in self._secondaries]),
        )

    def flux(self, t):
        """Compute the system flux at times ``t``.
        
        Args:
            t (scalar or vector): An array of times at which to evaluate
                the flux in units of :py:attr:`time_unit`.
        """
        return self.ops.flux(
            reshape(make_array_or_tensor(t), [-1]) * self._time_factor,
            self._primary._r,
            self._primary._m,
            self._primary._prot,
            self._primary._t0,
            self._primary._theta0,
            self._primary._map._L,
            self._primary._map._inc,
            self._primary._map._obl,
            self._primary._map._y,
            self._primary._map._u,
            self._primary._map._f,
            make_array_or_tensor([sec._r for sec in self._secondaries]),
            make_array_or_tensor([sec._m for sec in self._secondaries]),
            make_array_or_tensor([sec._prot for sec in self._secondaries]),
            make_array_or_tensor([sec._t0 for sec in self._secondaries]),
            make_array_or_tensor([sec._theta0 for sec in self._secondaries]),
            self._get_periods(),
            make_array_or_tensor([sec._ecc for sec in self._secondaries]),
            make_array_or_tensor([sec._w for sec in self._secondaries]),
            make_array_or_tensor([sec._Omega for sec in self._secondaries]),
            make_array_or_tensor([sec._inc for sec in self._secondaries]),
            make_array_or_tensor([sec._map._L for sec in self._secondaries]),
            make_array_or_tensor([sec._map._inc for sec in self._secondaries]),
            make_array_or_tensor([sec._map._obl for sec in self._secondaries]),
            make_array_or_tensor([sec._map._y for sec in self._secondaries]),
            make_array_or_tensor([sec._map._u for sec in self._secondaries]),
            make_array_or_tensor([sec._map._f for sec in self._secondaries]),
        )

    def position(self, t):
        """Compute the Cartesian positions of all bodies at times ``t``.
        
        Args:
            t (scalar or vector): An array of times at which to evaluate
                the position in units of :py:attr:`time_unit`.
        """
        x, y, z = self.ops.position(
            reshape(make_array_or_tensor(t), [-1]) * self._time_factor,
            self._primary._m,
            self._primary._t0,
            make_array_or_tensor([sec._m for sec in self._secondaries]),
            make_array_or_tensor([sec._t0 for sec in self._secondaries]),
            self._get_periods(),
            make_array_or_tensor([sec._ecc for sec in self._secondaries]),
            make_array_or_tensor([sec._w for sec in self._secondaries]),
            make_array_or_tensor([sec._Omega for sec in self._secondaries]),
            make_array_or_tensor([sec._inc for sec in self._secondaries]),
        )
        fac = np.reshape(
            [self._primary._length_factor]
            + [sec._length_factor for sec in self._secondaries],
            [-1, 1],
        )
        return (x / fac, y / fac, z / fac)

    def _get_periods(self):
        periods = [None for sec in self._secondaries]
        for i, sec in enumerate(self._secondaries):
            if sec.porb:
                periods[i] = sec.porb
            else:
                periods[i] = (
                    (2 * np.pi)
                    * sec._a ** (3 / 2)
                    / (math.sqrt(G_grav * (self._primary.m + sec._m)))
                )
        return make_array_or_tensor(periods)
