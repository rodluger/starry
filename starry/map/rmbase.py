# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["RMBase"]


class RMBase(object):
    """

    """
    def __init__(self, *args, **kwargs):
        super(RMBase, self).__init__(*args, **kwargs)
        self._filter_is_active = False
        self._alpha = 0.0

    def _compute_filter(self):
        """Set the filter coefficients to the RV field of the star."""
        cosi = np.cos(self.inc * np.pi / 180)
        sini = np.sin(self.inc * np.pi / 180)
        cosl = np.cos(self.obl * np.pi / 180)
        sinl = np.sin(self.obl * np.pi / 180)
        A = sini * cosl
        B = sini * sinl
        C = cosi
        ylm = np.array([0,
                        np.sqrt(3)*np.pi*B*(-A**2*self.alpha - B**2*self.alpha - C**2*self.alpha + 5)/15,
                        0,
                        np.sqrt(3)*np.pi*A*(-A**2*self.alpha - B**2*self.alpha - C**2*self.alpha + 5)/15,
                        0,
                        0,
                        0,
                        0,
                        0,
                        np.sqrt(70)*np.pi*B*self.alpha*(3*A**2 - B**2)/70,
                        2*np.sqrt(105)*np.pi*C*self.alpha*(-A**2 + B**2)/105,
                        np.sqrt(42)*np.pi*B*self.alpha*(A**2 + B**2 - 4*C**2)/210,
                        0,
                        np.sqrt(42)*np.pi*A*self.alpha*(A**2 + B**2 - 4*C**2)/210,
                        4*np.sqrt(105)*np.pi*A*B*C*self.alpha/105,
                        np.sqrt(70)*np.pi*A*self.alpha*(A**2 - 3*B**2)/70]) * (self.veq / np.pi)
        self._set_filter((slice(None, None, None), slice(None, None, None)), ylm)

    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, val):
        assert (val >= 0) and (val <= 1), "The rotational shear coefficient must be between 0 and 1."
        self._alpha = val

    def rv(self, *args, **kwargs):
        self._filter_is_active = True
        Iv = np.array(self.flux(*args, **kwargs))
        self._filter_is_active = False
        I = np.array(self.flux(*args, **kwargs))
        return Iv / I