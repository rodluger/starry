import oblate
import numpy as np
import pytest

# TODO!


class TestMath:
    @pytest.mark.parametrize("number", [1, 2, 3, 4])
    def test_division(self, number):
        assert number / number == 1
