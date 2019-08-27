# -*- coding: utf-8 -*-
import numpy as np

try:
    # starry requires exoplanet >= v0.2.0
    from packaging import version
    import exoplanet

    if version.parse(exoplanet.__version__) < version.parse("0.2.0"):
        exoplanet = None
except ModuleNotFoundError:
    exoplanet = None

# TODO
