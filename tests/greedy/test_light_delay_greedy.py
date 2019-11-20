# -*- coding: utf-8 -*-
"""Test light travel time delay"""
import starry
import numpy as np


def test_light_delay():
    pri = starry.Primary(starry.Map())
    sec = starry.Secondary(starry.Map(), porb=1.0)
    sys = starry.System(pri, sec, light_delay=True)
    assert sys.light_delay is True

    # TODO: Add tests here.
