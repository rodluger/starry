"""
Test current bugs/issues in starry.

"""
import starry
import pytest
import numpy as np


def test_segfault():
    
    # Settings
    maxl = 5
    nruns = 10
    yo = 0.3
    
    # Grids
    lgrid = np.arange(0, maxl, dtype=int)
    cgrid = np.array([30, 100], dtype=int)
    ggrid = np.array([False, True], dtype=bool)
    rgrid = np.array([0, 0.1])

    # Loop over degrees
    for i, lmax in enumerate(lgrid):
        map = starry.Map(ydeg=lmax, udeg=0)
        map[:, :] = 1

        # Loop over number of cadences
        for j, npts in enumerate(cgrid):
            theta = np.linspace(0, 360, npts)
            xo = np.linspace(-1.1, 1.1, npts)

            # Gradient on / off
            for k, gradient in enumerate(ggrid):

                # Occultations on / off
                for l, ro in enumerate(rgrid):

                    # Run several trials
                    for n in range(nruns):
                        
                        print(i, j, k, l, n)

                        map.flux(theta=theta, xo=xo, yo=yo, 
                                 ro=ro, gradient=gradient)


if __name__ == "__main__":
    test_segfault()