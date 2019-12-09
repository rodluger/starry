# -*- coding: utf-8 -*-
"""Test map rotation.

"""
import starry
import numpy as np


def test_rotate():
    map = starry.Map(1)
    map[1, 1] = 1
    map.rotate([0, 0, 1], 90.0)
    assert np.allclose(map.y.eval(), [1, 1, 0, 0])

    map = starry.Map(1)
    map[1, -1] = 1
    map.rotate([0, 0, 1], 90.0)
    assert np.allclose(map.y.eval(), [1, 0, 0, -1])


def test_rotate_spectral():
    map = starry.Map(1, nw=2)
    map[1, 1, 0] = 1
    map[1, -1, 1] = 1
    map.rotate([0, 0, 1], 90.0)
    assert np.allclose(map.y.eval(), [[1, 1], [1, 0], [0, 0], [0, -1]])
