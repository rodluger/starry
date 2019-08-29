# -*- coding: utf-8 -*-
"""Test Keplerian system stuff."""
import starry
import pytest
import theano
import numpy as np


def test_system():
    """
    TODO: This test needs to be improved.

    """
    A = starry.Primary(starry.Map())
    b = starry.Secondary(starry.Map(), porb=1.0, prot=1, r=0.1, L=0.1)
    sys = starry.System(A, b)
    assert np.allclose(sys.flux([0, 0.25, 0.5]), [1, 1.1, 1.09])
