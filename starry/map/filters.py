# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["FilterBase"]


class Filter(object):
    def __init__(self, parent):
        self.parent = parent
    def __getitem__(self, inds):
        return self.parent._get_filter(inds)
    def __setitem__(self, inds, vals):
        return self.parent._set_filter(inds, vals)
    def __repr__(self):
        return "<starry.Map.filter>"


class FilterBase(object):
    """

    """

    @property
    def filter(self):
        return Filter(self)

    def rv(self, *args, **kwargs):
        self._filter_is_active = False
        I = np.array(self.flux(*args, **kwargs))
        self._filter_is_active = True
        Iv = np.array(self.flux(*args, **kwargs))
        return Iv / I
