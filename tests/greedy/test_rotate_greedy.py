# -*- coding: utf-8 -*-
"""Test map rotation.

"""
import starry
import numpy as np


def test_rotate():
    map = starry.Map(1)
    map[1, 1] = 1
    map.rotate(np.array([0, 0, 1]), np.array(90.0))
    assert np.allclose(map.y, [1, 1, 0, 0])

    map = starry.Map(1)
    map[1, -1] = 1
    map.rotate(np.array([0, 0, 1]), np.array(90.0))
    assert np.allclose(map.y, [1, 0, 0, -1])


def test_rotate_spectral():
    map = starry.Map(1, nw=2)
    map[1, 1, 0] = 1
    map[1, -1, 1] = 1
    map.rotate(np.array([0, 0, 1]), np.array(90.0))
    assert np.allclose(map.y, [[1, 1], [1, 0], [0, 0], [0, -1]])
