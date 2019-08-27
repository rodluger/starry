# -*- coding: utf-8 -*-
from .maps import YlmBase, RVBase, ReflectedBase
import numpy as np
from astropy import units

try:
    # starry requires exoplanet >= v0.2.0
    from packaging import version
    import exoplanet

    if version.parse(exoplanet.__version__) < version.parse("0.2.0"):
        exoplanet = None
except ModuleNotFoundError:
    exoplanet = None


__all__ = ["KeplerianBase", "KeplerianSystem"]


class KeplerianBase(object):
    # TODO: Careful with `veq`!
    def __init__(self):
        # Attributes
        self._porb = 1.0

        # Units
        self.length_unit = units.Rsun
        self.time_unit = units.day

    @property
    def length_unit(self):
        """An ``astropy.units`` unit defining the length metric for this map."""
        return self._length_unit

    @length_unit.setter
    def length_unit(self, value):
        assert value.physical_type == "length"
        self._length_unit = value
        self._angle_factor = value.in_units(units.Rsun)

    @property
    def time_unit(self):
        """An ``astropy.units`` unit defining the time metric for this map."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, value):
        assert value.physical_type == "time"
        self._time_unit = value
        self._angle_factor = value.in_units(units.day)

    @property
    def porb(self):
        """The orbital period in units of :py:attr:`time_unit`."""
        return self._porb / self.time_unit

    @porb.setter
    def porb(self, value):
        self._porb = value * self.time_unit


class KeplerianSystem(object):
    def __init__(self):
        # Require exoplanet
        assert exoplanet is not None, "This class requires exoplanet >= 0.2.0."


def Body():
    # TODO: Class factory
    pass
