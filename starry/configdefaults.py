# -*- coding: utf-8 -*-
import logging

rootLogger = logging.getLogger("starry")
rootLogger.addHandler(logging.StreamHandler())


class Config(object):
    """Global config container.

    Users should access this as :py:obj:`starry.config`.

    """

    def __init__(self):
        self._allow_changes = True
        self.lazy = True
        self.quiet = False
        self.profile = False

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

    @property
    def profile(self):
        """Enable function profiling in lazy mode."""
        return self._profile

    @quiet.setter
    def quiet(self, value):
        self._quiet = value
        if self._quiet:
            self.rootLogger.setLevel(logging.ERROR)
        else:
            self.rootLogger.setLevel(logging.INFO)

    @lazy.setter
    def lazy(self, value):
        if (self._allow_changes) or (self._lazy == value):
            self._lazy = value
        else:
            raise Exception(
                "Cannot change the `starry` config at this time. "
                "Config options should be set before instantiating any `starry` maps."
            )

    @profile.setter
    def profile(self, value):
        if (self._allow_changes) or (self._profile == value):
            self._profile = value
        else:
            raise Exception(
                "Cannot change the `starry` config at this time. "
                "Config options should be set before instantiating any `starry` maps."
            )

    def freeze(self):
        self._allow_changes = False
