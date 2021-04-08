import oblate
import numpy as np
import pytest


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


@pytest.mark.parametrize("kwargs", [dict(bo=0.58, ro=0.4, f=0.2, theta=0.5)])
def test_ksq_gt_one(solvers, kwargs):
    """
    Test occultations with k^2 > 1.

    """
    numsT = solvers["num"].get_sT(**kwargs)
    pysT = solvers["py"].get_sT(**kwargs)
    cppsT = solvers["cpp"].get_sT(**kwargs)
    assert np.allclose(numsT, pysT, atol=0.05)
    assert np.allclose(pysT, cppsT)


@pytest.mark.xfail(reason="Complex intermediate values when computing J.")
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.9, ro=0.1, f=0.2, theta=0.8),
        dict(bo=0.65, ro=0.4, f=0.2, theta=0.5),
    ],
)
def test_ksq_lt_one(solvers, kwargs):
    """
    Test occultations with k^2 < 1.

    """
    pysT = solvers["py"].get_sT(**kwargs)
    cppsT = solvers["cpp"].get_sT(**kwargs)
    numsT = solvers["num"].get_sT(**kwargs)
    assert np.allclose(numsT, pysT, atol=0.05)
    assert np.allclose(pysT, cppsT)


@pytest.mark.parametrize("kwargs", [dict(bo=0.3, ro=0.8, f=0.4, theta=0.5)])
def test_compare_numerical_to_brute(kwargs):
    """

    """
    brute = oblate.BruteSolver(5).get_sT(**kwargs).flatten()
    num = oblate.NumericalSolver(5).get_sT(**kwargs).flatten()
    assert np.all(np.abs(brute - num) < 0.001)
    assert np.all(np.abs((brute - num) / num) < 0.002)

