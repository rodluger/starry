"""
Test current bugs/issues in starry.

"""
import starry
import pytest
import numpy as np


@pytest.mark.xfail
def test_axis():
    map = starry.Map()
    map.axis = [0, 0, 1]
    assert np.allclose(map.axis, [0, 0, 1])