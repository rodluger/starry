# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from starry.kepler import Primary, Secondary, System


def get_system(n_sec=1):
    primary = Primary()
    secondaries = []
    for i in range(n_sec):
        secondaries.append(Secondary())
    return System(primary, *secondaries)


def test_memory():
    system = get_system()
    system.compute(np.linspace(0, 5, 10))

    system = get_system(2)
    system.compute(np.linspace(0, 5, 10))
