# -*- coding: utf-8 -*-
import theano.tensor as tt


class Config(object):
    def __init__(self):
        self._lazy = True
        self._allow_changes = True

    @property
    def lazy(self):
        """Indicates whether or not the map evaluates things lazily.

        If True, all attributes and method return values are unevaluated 
        ``theano`` nodes. This is particularly useful for model building and 
        integration with ``pymc3``. In lazy mode, call the ``.eval()`` method 
        on any ``theano`` node to compute and return its numerical value. 

        If False, ``starry`` will automatically compile methods called by the 
        user, and all methods will return numerical values as in the previous 
        version of the code.
        """
        return self._lazy

    @lazy.setter
    def lazy(self, value):
        if (self._lazy == value) or self._allow_changes:
            self._lazy = value
        else:
            raise Exception(
                "Cannot change the `starry` config at this time. "
                "Config options should be set before instantiating any `starry` maps."
            )

    def freeze(self):
        self._allow_changes = False
