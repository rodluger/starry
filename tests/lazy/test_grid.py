# -*- coding: utf-8 -*-
"""

"""
import starry
import pytest
import theano.tensor as tt
import itertools


res = [30, 50]
compile = [True, False]
params = itertools.product(res, compile)


@pytest.mark.parametrize("res,compile", params)
def test_ortho_grid(res, compile):
    map = starry.Map(1)
    if compile:
        x, y, z = map.ops.compute_ortho_grid(res)
    else:
        x, y, z = map.ops.compute_ortho_grid(tt.as_tensor_variable(res)).eval()
    assert len(x) == res ** 2
