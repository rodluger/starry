import oblate
import numpy as np
import pytest


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


@pytest.mark.parametrize("kwargs", [dict(bo=0.58, ro=0.4, f=0.2, theta=0.5)])
def test_compare_to_python_ksq_gt_one(solvers, kwargs):
    """
    Compare C++ implementation to Python implementation for k^2 > 1.

    """
    cppsT = solvers["cpp"].get_sT(**kwargs)
    pysT = solvers["py"].get_sT(**kwargs)
    assert np.allclose(cppsT, pysT)


@pytest.mark.xfail(
    reason="Getting complex intermediate values when computing J."
)
@pytest.mark.parametrize("kwargs", [dict(bo=0.65, ro=0.4, f=0.2, theta=0.5)])
def test_compare_to_python_ksq_lt_one(solvers, kwargs):
    """
    Compare C++ implementation to Python implementation for k^2 < 1.

    """
    cppsT = solvers["cpp"].get_sT(**kwargs)
    pysT = solvers["py"].get_sT(**kwargs)
    assert np.allclose(cppsT, pysT)


@pytest.mark.parametrize("kwargs", [dict(bo=0.3, ro=0.8, f=0.4, theta=0.5)])
def test_compare_numerical_to_brute(kwargs):
    """

    """
    brute = oblate.BruteSolver(5).get_sT(**kwargs).flatten()
    num = oblate.NumericalSolver(5).get_sT(**kwargs).flatten()
    assert np.all(np.abs(brute - num) < 0.001)
    assert np.all(np.abs((brute - num) / num) < 0.002)
