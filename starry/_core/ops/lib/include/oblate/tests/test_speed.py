import oblate
import numpy as np
import time
import os
import pytest


@pytest.mark.skipif(
    os.getenv("CI", "false") == "false", reason="Only run this on CI."
)
def test_speed(solvers, nruns=1000, niter=30):
    """
    Compare the speed of the C++ and Python implementations.

    """
    # Run the timing tests a bunch of times
    cpptime = np.zeros(niter)
    cppxtime = np.zeros(niter)
    pytime = np.zeros(niter)
    for k in range(niter):

        # Time the C++ solver (linear)
        tstart = time.time()
        res = solvers["cpp"].get_sT(nruns=nruns)
        cpptime[k] = (time.time() - tstart) / nruns

        # Time the C++ solver (exact)
        tstart = time.time()
        res = solvers["cppx"].get_sT(nruns=nruns)
        cppxtime[k] = (time.time() - tstart) / nruns

        # Time the Python solver
        tstart = time.time()
        res = solvers["py"].get_sT(nruns=nruns)
        pytime[k] = (time.time() - tstart) / nruns

    # Log
    print("")
    print(
        "C++ (linear):    {:.3} +/- {:.3} ms".format(
            np.median(cppxtime) * 1e3, np.std(cppxtime) * 1e3
        )
    )
    print(
        "C++ (exact):     {:.3} +/- {:.3} ms".format(
            np.median(cpptime) * 1e3, np.std(cpptime) * 1e3
        )
    )
    print(
        "Python (linear): {:.3} +/- {:.3} ms".format(
            np.median(pytime) * 1e3, np.std(pytime) * 1e3
        )
    )
