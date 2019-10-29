# -*- coding: utf-8 -*-
import logging

rootLogger = logging.getLogger()
rootLogger.addHandler(logging.StreamHandler())


class Config(object):
    def __init__(self):
        self._allow_lazy_changes = True
        self.lazy = True
        self.quiet = False

    @property
    def rootLogger(self):
        return rootLogger

    @property
    def rootHandler(self):
        return rootLogger.handlers[0]

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

    @property
    def quiet(self):
        """Indicates whether or not to suppress informational messages."""
        return self._quiet

    @quiet.setter
    def quiet(self, value):
        self._quiet = value
        if self._quiet:
            rootLogger.setLevel(logging.ERROR)
        else:
            rootLogger.setLevel(logging.INFO)

    @lazy.setter
    def lazy(self, value):
        if (self._allow_lazy_changes) or (self._lazy == value):
            self._lazy = value
        else:
            raise Exception(
                "Cannot change the `starry` config at this time. "
                "Config options should be set before instantiating any `starry` maps."
            )

    def freeze(self):
        self._allow_lazy_changes = False
