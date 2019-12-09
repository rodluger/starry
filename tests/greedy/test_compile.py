# -*- coding: utf-8 -*-
"""Test theano compiling."""
import starry
import pytest
import theano
import numpy as np


def test_test_value_error():
    """
    Ensure that no error is raised when there
    are missing test values in our compiled
    functions, as pymc3 forces theano errors when
    test values are missing.
    
    """
    theano.config.compute_test_value = "raise"
    map = starry.Map()
    flux = map.flux()
    assert theano.config.compute_test_value == "raise"
