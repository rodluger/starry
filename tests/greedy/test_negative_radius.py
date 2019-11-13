# -*- coding: utf-8 -*-
"""
Test behavior for negative occultor radii.

"""
import starry


def test_negative_radius():
    map = starry.Map(ydeg=2)
    map[1, 0] = 1.0
    try:
        flux = map.flux(xo=0.99, yo=0, zo=1, ro=-1e-15)
    except RuntimeError as e:
        assert "Occultor radius is negative. Aborting." in str(e)
