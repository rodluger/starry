# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["MapSum"]


class MapSum(object):
    """

    """

    def __init__(self, map):
        """

        """
        self.maps = [map]
        self._multi = map._multi
        self._temporal = map._temporal
        self._spectral = map._spectral
        self._reflected = map._reflected
        self._ydeg = map.ydeg
        self._udeg = map.udeg
        self._nw = map.nw
        self._nt = map.nt

    @property
    def ydeg(self):
        return self._ydeg

    @property
    def udeg(self):
        return self._udeg

    @property
    def nt(self):
        return self._nt

    @property
    def nw(self):
        return self._nw

    def __add__(self, other):
        """

        """
        # Ensure the maps have the same type
        assert (self._multi == other._multi and
                self._temporal == other._temporal and
                self._spectral == other._spectral and
                self._reflected == other._reflected), \
                    "Map addition is only supported for maps of the same type."

        # Ensure they have the same dimensions
        if self._spectral:
            assert self.nw == other.nw, \
                "Maps must have the same number of wavelength bins."
        elif self._temporal:
            assert self.nt == other.nt, \
                "Maps must have the same number of temporal bins."

        # Figure out the degree of the map
        self._ydeg = max(self.ydeg, other.ydeg)
        self._udeg = max(self.udeg, other.udeg)

        # Add it to the list
        self.maps += [other]
        return self
    
    def __repr__(self):
        """

        """
        return " + ".join([map.__repr__() for map in self.maps])
    
    def _reshape_theta(self, theta):
        """

        """
        theta = np.atleast_1d(theta)
        if len(theta.shape) == 1:
            if theta.shape == (1,):
                theta = np.array([theta[0] for n in range(len(self.maps))])
            else:
                assert theta.shape[0] == len(self.maps), \
                    "Incorrect dimensions for `theta`." 
        elif len(theta.shape) == 2:
            assert theta.shape[0] == len(self.maps), \
                "Incorrect dimensions for `theta`."
        else:
            raise ValueError("Incorrect dimensions for `theta`.")
        return theta

    def linear_intensity_model(self, theta=0, **kwargs):
        """
        
        """
        theta = self._reshape_theta(theta)
        return np.hstack([self.maps[n].linear_intensity_model(theta=theta[n], **kwargs)
                          for n in range(len(self.maps))])

    def render(self, theta=0, **kwargs):
        """

        """
        theta = self._reshape_theta(theta)
        kwargs["rotate_if_rect"] = True
        return np.sum([self.maps[n].render(theta=theta[n] - theta[0], **kwargs) 
                       for n in range(len(self.maps))], axis=0)
    
    def show(self, **kwargs):
        """

        """
        Z = self.render(**kwargs)
        return self.maps[0].show(Z=Z, **kwargs)

    def linear_flux_model(self, theta=0, **kwargs):
        """
        .. todo:: gradients!
        """
        theta = self._reshape_theta(theta)
        return np.hstack([self.maps[n].linear_flux_model(theta=theta[n], **kwargs)
                          for n in range(len(self.maps))])

    def flux(self, theta=0, **kwargs):
        """
        .. todo:: gradients!
        """
        theta = self._reshape_theta(theta)
        return np.sum([self.maps[n].flux(theta=theta[n], **kwargs) 
                       for n in range(len(self.maps))], axis=0)