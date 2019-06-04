# -*- coding: utf-8 -*-
"""Test rotation axis i/o."""
import starry
import pytest
import numpy as np


# Instantiate a global map
map = starry.Map(1, lazy=False)


@pytest.fixture(
    scope="class",
    params=[
        [[1, 0, 0]],
        [[0, 1, 0]],
        [[0, 0, 1]],
        [[-1, 0, 0]],
        [[0, -1, 0]],
        [[0, 0, -1]],
        [[np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]],
        [[-np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]],
        [[np.sqrt(1 / 3), -np.sqrt(1 / 3), np.sqrt(1 / 3)]],
        [[-np.sqrt(1 / 3), np.sqrt(1 / 3), -np.sqrt(1 / 3)]],
        [[-np.sqrt(1 / 3), -np.sqrt(1 / 3), np.sqrt(1 / 3)]],
        [[-np.sqrt(1 / 3), np.sqrt(1 / 3), -np.sqrt(1 / 3)]],
        [[np.sqrt(1 / 3), -np.sqrt(1 / 3), -np.sqrt(1 / 3)]],
        [[-np.sqrt(1 / 3), -np.sqrt(1 / 3), -np.sqrt(1 / 3)]],
        [[np.sqrt(1 / 2), np.sqrt(1 / 2), 0]],
        [[-np.sqrt(1 / 2), np.sqrt(1 / 2), 0]],
        [[np.sqrt(1 / 2), -np.sqrt(1 / 2), 0]],
        [[-np.sqrt(1 / 2), -np.sqrt(1 / 2), 0]],
        [[np.sqrt(1 / 2), 0, np.sqrt(1 / 2)]],
        [[-np.sqrt(1 / 2), 0, np.sqrt(1 / 2)]],
        [[np.sqrt(1 / 2), 0, -np.sqrt(1 / 2)]],
        [[-np.sqrt(1 / 2), 0, -np.sqrt(1 / 2)]],
        [[0, np.sqrt(1 / 2), np.sqrt(1 / 2)]],
        [[0, -np.sqrt(1 / 2), np.sqrt(1 / 2)]],
        [[0, np.sqrt(1 / 2), -np.sqrt(1 / 2)]],
        [[0, -np.sqrt(1 / 2), -np.sqrt(1 / 2)]],
    ]
)
def axis(request):
    axis, = request.param
    return axis

class TestAxis:
    """Test setting the axis of rotation."""

    def test_axis(self, axis):
        map.axis = axis
        assert np.allclose(map.axis, axis)