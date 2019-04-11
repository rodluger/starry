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