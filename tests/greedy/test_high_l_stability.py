import starry
import matplotlib.pyplot as plt
import numpy as np
import pytest


# BROKEN. We need to re-parametrize
# the L and K integrals (or compute them numerically).
@pytest.mark.xfail
def test_high_l_stability(plot=False):
    map = starry.Map(ydeg=20, reflected=False)
    map[1:, :] = 1
    xo = np.logspace(-2, np.log10(2.0), 1000)
    yo = 0
    ro = 0.9
    flux = map.flux(xo=xo, yo=yo, ro=ro)
    bo = np.sqrt(xo ** 2 + yo ** 2)
    ksq = (1 - ro ** 2 - bo ** 2 + 2 * bo * ro) / (4 * bo * ro)

    if plot:
        plt.plot(ksq, flux)
        plt.xscale("log")
        plt.show()

    # Check for gross stability issues here
    assert np.std(flux[ksq > 1]) < 0.1


if __name__ == "__main__":
    starry.config.lazy = False
    plt.style.use("default")
    test_high_l_stability(plot=True)
