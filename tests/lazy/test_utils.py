# -*- coding: utf-8 -*-
"""Test Op utils.

"""
import theano.tensor as tt
from starry._core import math
from scipy.linalg import block_diag as scipy_block_diag
import numpy as np


def test_block_diag():
    C1 = np.ones((2, 2))
    C2 = np.ones((3, 3)) * 2
    C3 = np.ones((3, 3)) * 3
    C = scipy_block_diag(C1, C2, C3)

    C1 = tt.as_tensor_variable(np.ones((2, 2)))
    C2 = tt.as_tensor_variable(np.ones((3, 3)) * 2)
    C3 = tt.as_tensor_variable(np.ones((3, 3)) * 3)

    assert np.allclose(C, math.block_diag(C1, C2, C3).eval())
