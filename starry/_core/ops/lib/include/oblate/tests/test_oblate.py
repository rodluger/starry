import oblate
import numpy as np
import matplotlib.pyplot as plt
import time


def test_compare_to_numerical(lmax=8):

    # TODO: C++ version can't currently handle f < 1e-8

    cppsolver = oblate.CppSolver(lmax)
    numsolver = oblate.NumericalSolver(lmax)

    # Check that they agree for f = 0
    cppsT = cppsolver.get_sT(f=0.0)
    numsT = numsolver.get_sT(f=0.0)
    assert np.allclose(cppsT, numsT)

    # Check that the derivatives wrt f match
    df = 1e-8
    cppdsTdf = (cppsolver.get_sT(f=df) - cppsT) / df
    numdsTdf = (numsolver.get_sT(f=df) - numsT) / df
    assert np.allclose(cppdsTdf, numdsTdf)


def test_compare_to_python_ksq_gt_one(lmax=8):

    cppsolver = oblate.CppSolver(lmax)
    pysolver = oblate.PythonSolver(lmax)

    cppsT = cppsolver.get_sT(bo=0.58, ro=0.4, f=0.2, theta=0.5)
    pysT = pysolver.get_sT(bo=0.58, ro=0.4, f=0.2, theta=0.5)
    assert np.allclose(cppsT, pysT)


def test_compare_to_python_ksq_lt_one(lmax=8):

    # TODO: Getting NaNs here!

    cppsolver = oblate.CppSolver(lmax)
    pysolver = oblate.PythonSolver(lmax)

    cppsT = cppsolver.get_sT(bo=0.65, ro=0.4, f=0.2, theta=0.5)
    pysT = pysolver.get_sT(bo=0.65, ro=0.4, f=0.2, theta=0.5)
    assert np.allclose(cppsT, pysT)


def test_speed(lmax=8, nruns=1000, niter=30):
    cppsolver = oblate.CppSolver(lmax)
    pysolver = oblate.PythonSolver(lmax)

    cpptime = np.zeros(niter)
    pytime = np.zeros(niter)
    for k in range(niter):

        tstart = time.time()
        res = cppsolver.get_sT(nruns=nruns)
        cpptime[k] = (time.time() - tstart) / nruns

        tstart = time.time()
        res = pysolver.get_sT(nruns=nruns)
        pytime[k] = (time.time() - tstart) / nruns

    print(
        "C++:    {:.3} +/- {:.3} ms".format(
            np.median(cpptime) * 1e3, np.std(cpptime) * 1e3
        )
    )
    print(
        "Python: {:.3} +/- {:.3} ms".format(
            np.median(pytime) * 1e3, np.std(pytime) * 1e3
        )
    )
