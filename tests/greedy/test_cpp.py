# -*- coding: utf-8 -*-
"""
Tests for C++ functions. TODO.

"""
from starry import _c_ops as Ops
import numpy as np
import pytest


# Only run tests if `STARRY_UNIT_TESTS=1` on compile
cpp = pytest.mark.skipif(
    not Ops.STARRY_UNIT_TESTS, reason="c++ unit tests not found"
)
