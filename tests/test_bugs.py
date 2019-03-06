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
def test_z_rotation():
    # axis = [0, 0, 1] and axis = [0, 0, -1] give the same rotation
    raise Exception("")

@pytest.mark.xfail
def test_show_reflected():
    # Things go to NaN for some source positions
    map1 = starry.Map(ydeg=2, udeg=0, reflected=True)
    map1[0, 0] = 1
    Z = map1.render(projection="ortho", res=75, source=[1, 2, 3])
    assert len(np.where(np.isfinite(Z))[0]) > 0