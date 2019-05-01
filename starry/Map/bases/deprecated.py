# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["DeprecationBase"]


class DeprecationBase(object):
    """

    """

    def load_image(self, *args, **kwargs):
        raise NameError("This method is deprecated." +
                        "Please use the `load` method instead.")

    def animate(self, *args, **kwargs):
        raise NameError("This method is deprecated." +
                        "Please use the `show` method instead.")
    
    def load_healpix(self, *args, **kwargs):
        raise NameError("This method is deprecated." +
                        "Please use the `load` method instead.")
    
    def add_gaussian(self, *args, **kwargs):
        raise NameError("This method is deprecated." +
                        "Please use the `add_spot` method instead.")
    
    @property
    def lmax(self):
        raise NameError("This attribute is deprecated. Use `ydeg` instead.")

    @property
    def nwav(self):
        raise NameError("This attribute is deprecated. Use `nw` instead.")