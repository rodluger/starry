# -*- coding: utf-8 -*-
"""
There's a bug in tt.mgrid that causes different
behavior whether it's compiled or not. We implemented
some hacks in `starry` to circumvent this.
See docstring of `compute_ortho_grid` in "core.py"

"""
import starry
import pytest
from starry.compat import tt
import itertools
import numpy as np


@pytest.mark.parametrize("compile", [True, False])
def test_ortho_grid(compile):
    map = starry.Map(1)
    for res in np.arange(30, 101):
        if compile:
            (lat, lon), (x, y, z) = map.ops.compute_ortho_grid(res)
        else:
            latlon, xyz = map.ops.compute_ortho_grid(
                tt.as_tensor_variable(res)
            )
            x, y, z = xyz.eval()
        assert len(x) == res ** 2
