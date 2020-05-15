# -*- coding: utf-8 -*-
"""
There's a bug in tt.mgrid that causes different
behavior whether it's compiled or not. We implemented
some hacks in `starry` to circumvent this.
See docstring of `compute_ortho_grid` in "core.py"

"""
import starry
import pytest
import theano.tensor as tt
import itertools
import numpy as np


res = np.arange(30, 101)
compile = [True, False]
params = itertools.product(res, compile)
map = starry.Map(1)


@pytest.mark.parametrize("res,compile", params)
def test_ortho_grid(res, compile):
    if compile:
        x, y, z = map.ops.compute_ortho_grid(res)
    else:
        x, y, z = map.ops.compute_ortho_grid(tt.as_tensor_variable(res)).eval()
    assert len(x) == res ** 2
