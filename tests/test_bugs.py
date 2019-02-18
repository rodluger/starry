"""
Test current bugs/issues in starry.

"""
import starry
import pytest
import numpy as np


@pytest.mark.xfail
def test_bad_lm():
    map = starry.Map(5)
    try:
        map[3, 4] = 1.0
        error_thrown = False
    except:
        error_thrown = True
    assert error_thrown


@pytest.mark.xfail
def test_load_image():
    # todo
    raise Exception("Wrong orientation of maps with `load_image`.")