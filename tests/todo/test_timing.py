"""Test evaluation times compared to the beta version."""
import numpy as np
import starry
import starry_beta
import pytest
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def test_time_default():
    # ***Currently segfaulting!!!!!***

    # Settings
    maxl = 15
    nruns = 10
    yo = 0.3
    
    # Grids
    lgrid = np.arange(0, maxl, dtype=int)
    cgrid = np.array([1, 10, 30, 100, 300, 1000, 3000], dtype=int)
    ggrid = np.array([False, True], dtype=bool)
    rgrid = np.array([0, 0.1])
    vbeta = np.empty((lgrid.shape[0], cgrid.shape[0], 
                      ggrid.shape[0], rgrid.shape[0]))
    v1 = np.empty((lgrid.shape[0], cgrid.shape[0], 
                   ggrid.shape[0], rgrid.shape[0]))
    
    # Loop over degrees
    for i, lmax in tqdm(enumerate(lgrid), total=lgrid.shape[0]):
        map_vbeta = starry_beta.Map(lmax=lmax)
        map_vbeta[:, :] = 1
        map_v1 = starry.Map(ydeg=lmax, udeg=0)
        map_v1[:, :] = 1

        # Loop over number of cadences
        for j, npts in enumerate(cgrid):
            theta = np.linspace(0, 360, npts)
            xo = np.linspace(-1.1, 1.1, npts)

            # Gradient on / off
            for k, gradient in enumerate(ggrid):

                # Occultations on / off
                for l, ro in enumerate(rgrid):
                    tvbeta = np.empty(nruns, dtype=float)
                    tv1 = np.empty(nruns, dtype=float)
                    
                    # Run several trials
                    for n in range(nruns):

                        # beta
                        t1 = time.time()
                        res = map_vbeta.flux(theta=theta, xo=xo, yo=yo, 
                                             ro=ro, gradient=gradient)
                        tvbeta[n] = time.time() - t1

                        # v1
                        t1 = time.time()
                        res = map_v1.flux(theta=theta, xo=xo, yo=yo, 
                                             ro=ro, gradient=gradient)
                        tv1[n] = time.time() - t1
                    
                    # Take the median
                    vbeta[i, j, k, l] = np.median(tvbeta)
                    v1[i, j, k, l] = np.median(tv1)


if __name__ == "__main__":
    test_time_default()