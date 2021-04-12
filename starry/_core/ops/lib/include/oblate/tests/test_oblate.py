import oblate
import numpy as np
import pytest
import itertools


class Compare:
    @pytest.mark.parametrize(
        "compare",
        [
            ("cpp", "py"),
            ("cpp", "lin"),
            ("cpp", "num"),
            ("cpp", "brute"),
            ("brute", "num"),
        ],
    )
    def test_compare(self, solvers, kwargs, compare):
        kwargs = dict(kwargs)
        brute_error = 1e-3
        max_error = kwargs.pop("max_error", 1e-8)
        atol = {
            ("cpp", "py"): 1e-12,
            ("cpp", "lin"): 1e-12,
            ("cpp", "num"): max_error,
            ("cpp", "brute"): max(max_error, brute_error),
            ("brute", "num"): brute_error,
        }
        assert np.allclose(
            solvers[compare[0]].get_sT(**kwargs),
            solvers[compare[1]].get_sT(**kwargs),
            atol=atol[compare],
        )


@pytest.mark.parametrize(
    "kwargs", [dict(bo=0.58, ro=0.4, f=0.2, theta=0.5, max_error=0.05)]
)
class TestKsqGreaterThanOne(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.91, ro=0.1, f=0.2, theta=0.8, max_error=0.05),
        dict(bo=0.65, ro=0.4, f=0.2, theta=0.5, max_error=0.05),
    ],
)
class TestKsqLessThanOne(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.95, ro=0.1, f=0.0, theta=0.8, max_error=1e-12),
        dict(bo=0.95, ro=0.1, f=1e-15, theta=0.8, max_error=1e-12),
        dict(bo=0.95, ro=0.1, f=1e-12, theta=0.8, max_error=1e-12),
    ],
)
class TestNearlySpherical(Compare):
    pass
