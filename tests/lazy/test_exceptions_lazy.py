# -*- coding: utf-8 -*-
"""
Test various exceptions throughout the code.

"""
import starry
import pytest


def test_lazy_change():
    map = starry.Map()
    with pytest.raises(Exception) as e:
        starry.config.lazy = False
    assert "Cannot change the `starry` config at this time." in str(e.value)
