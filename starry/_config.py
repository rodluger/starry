# -*- coding: utf-8 -*-
import logging

rootLogger = logging.getLogger("starry")
rootLogger.addHandler(logging.StreamHandler())
rootLogger.setLevel(logging.INFO)


class ConfigType(type):
    """Global config container."""

    @property
    def rootLogger(cls):
        return rootLogger

    @property
    def rootHandler(cls):
        return rootLogger.handlers[0]

    @property
    def lazy(cls):
        """Indicates whether or not the map evaluates things lazily.

        If True, all attributes and method return values are unevaluated
        ``theano`` nodes. This is particularly useful for model building and
        integration with ``pymc3``. In lazy mode, call the ``.eval()`` method
        on any ``theano`` node to compute and return its numerical value.

        If False, ``starry`` will automatically compile methods called by the
        user, and all methods will return numerical values as in the previous
        version of the code.
        """
        return cls._lazy

    @property
    def quiet(cls):
        """Indicates whether or not to suppress informational messages."""
        return cls._quiet

    @property
    def mode(cls):
        """Enable function profiling in lazy mode."""
        return cls._mode

    @property
    def profile(cls):
        """Enable function profiling in lazy mode."""
        return cls._profile

    @quiet.setter
    def quiet(cls, value):
        cls._quiet = value
        if cls._quiet:
            cls.rootLogger.setLevel(logging.ERROR)
        else:
            cls.rootLogger.setLevel(logging.INFO)

    @lazy.setter
    def lazy(cls, value):
        if (cls._allow_changes) or (cls._lazy == value):
            cls._lazy = value
        else:
            raise Exception(
                "Cannot change the `starry` config at this time. "
                "Config options should be set before instantiating any `starry` maps."
            )

    @profile.setter
    def profile(cls, value):
        if (cls._allow_changes) or (cls._profile == value):
            cls._profile = value
        else:
            raise Exception(
                "Cannot change the `starry` config at this time. "
                "Config options should be set before instantiating any `starry` maps."
            )

    @mode.setter
    def mode(cls, value):
        if (cls._allow_changes) or (cls._mode == value):
            cls._mode = value
        else:
            raise Exception(
                "Cannot change the `starry` config at this time. "
                "Config options should be set before instantiating any `starry` maps."
            )

    def freeze(cls):
        cls._allow_changes = False


class config(metaclass=ConfigType):
    _allow_changes = True
    _lazy = True
    _quiet = False
    _profile = False
    _mode = None
