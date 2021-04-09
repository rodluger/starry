import oblate
import numpy as np
import pytest
import itertools


@pytest.mark.xfail(reason="hyp2f1(-0.5, 21, 22, 0.999987) doesn't converge.")
def test_hyp2f1_convergence(solvers):
    solvers["cpp"].get_sT(bo=0.93, ro=0.1, f=1e-6, theta=0.5)


@pytest.mark.xfail(reason="The C++ version can't currently handle f = 0.")
@pytest.mark.parametrize("kwargs", [dict(bo=0.58, ro=0.4, theta=0.5)])
def test_compare_to_numerical(solvers, kwargs):
    """
    Compare C++ implementation to numerical solution of Green's theorem.

    """
    # Check that they agree for f = 0
    cppsT = solvers["cpp"].get_sT(f=0.0, **kwargs)
    numsT = solvers["num"].get_sT(f=0.0, **kwargs)
    assert np.allclose(cppsT, numsT)

    # Check that the derivatives wrt f match
    df = 1e-8
    cppdsTdf = (solvers["cpp"].get_sT(f=df, **kwargs) - cppsT) / df
    numdsTdf = (solvers["num"].get_sT(f=df, **kwargs) - numsT) / df
    assert np.allclose(cppdsTdf, numdsTdf)


class Compare:
    @pytest.mark.parametrize(
        "compare",
        [
            ("cpp", "py"),
            ("num", "py"),
            ("cpp", "num"),
            ("brute", "num"),
            ("brute", "cpp"),
            ("brute", "py"),
        ],
    )
    def test_compare(self, solvers, kwargs, compare):
        kwargs = dict(kwargs)
        brute_error = 1e-3
        max_error = kwargs.pop("max_error", 1e-8)
        atol = {
            ("cpp", "py"): 1e-12,
            ("num", "py"): max_error,
            ("cpp", "num"): max_error,
            ("brute", "num"): brute_error,
            ("brute", "cpp"): max(max_error, brute_error),
            ("brute", "py"): max(max_error, brute_error),
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
