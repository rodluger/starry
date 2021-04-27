import starry
import numpy as np
import pytest
from scipy.integrate import dblquad


class StarrySolver:
    """
    Wrapper for the `sT` solution vector computed by `starry`.

    """

    def __init__(self, lmax=5):
        self.map = starry.Map(lmax, oblate=True, gdeg=0)

    def get_sT(self, bo=0.58, ro=0.4, f=0.2, theta=0.5, **kwargs):
        return self.map.ops.sT(
            np.array(f, dtype="float64"),
            np.array([theta], dtype="float64"),
            np.array([bo], dtype="float64"),
            np.array(ro, dtype="float64"),
        ).flatten()


class BruteSolver:
    """
    Brute-force grid integration for computing the `sT` solution vector.

    """

    def __init__(self, lmax=5):
        self.lmax = lmax

        # Integrate the Green's basis over
        # the entire unit disk
        self.sT0 = np.zeros((self.lmax + 1) ** 2)
        for n in range((self.lmax + 1) ** 2):
            self.sT0[n], _ = dblquad(
                lambda x, y: self.g(n, x, y),
                -1,
                1,
                lambda x: -np.sqrt(1 - x ** 2),
                lambda x: np.sqrt(1 - x ** 2),
            )

    def g(self, n, x, y, z=None):
        """
        Return the nth term of the Green's basis.

        """
        if z is None:
            z2 = 1 - x ** 2 - y ** 2
            z = np.sqrt(np.abs(z2))
            on_star = z2 > 0
            if hasattr(z, "__len__"):
                z[~on_star] = np.nan
            else:
                if not on_star:
                    z2 = np.nan
        l = int(np.floor(np.sqrt(n)))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            I = [mu // 2]
            J = [nu // 2]
            K = [0]
            C = [(mu + 2) // 2]
        elif (l == 1) and (m == 0):
            I = [0]
            J = [0]
            K = [1]
            C = [1]
        elif (mu == 1) and (l % 2 == 0):
            I = [l - 2]
            J = [1]
            K = [1]
            C = [3]
        elif mu == 1:
            I = [l - 3, l - 1, l - 3]
            J = [0, 0, 2]
            K = [1, 1, 1]
            C = [-1, 1, 4]
        else:
            I = [(mu - 5) // 2, (mu - 5) // 2, (mu - 1) // 2]
            J = [(nu - 1) // 2, (nu + 3) // 2, (nu - 1) // 2]
            K = [1, 1, 1]
            C = [(mu - 3) // 2, -(mu - 3) // 2, -(mu + 3) // 2]
        res = z * 0
        for i, j, k, c in zip(I, J, K, C):
            if c != 0:
                res += c * x ** i * y ** j * z ** k
        return res

    def get_sT(self, bo=0.58, ro=0.4, f=0.2, theta=0.5, res=999, **kwargs):

        # Grid up the occultor
        xo = bo * np.sin(theta)
        yo = bo * np.cos(theta)
        x = np.linspace(xo - ro, xo + ro, res)
        y = np.linspace(yo - ro, yo + ro, res)
        x, y = np.meshgrid(x, y)
        b = 1 - f
        on_star = 1 - x ** 2 - (y / b) ** 2 > 0

        # Compute the Green's basis in (x, y') where
        #
        #     y' = y / b
        #
        # is the transformed y coordinate
        g = np.array(
            [self.g(n, x, y / b) for n in range((self.lmax + 1) ** 2)]
        )

        # Compute the masked pixels
        under_occultor = (x - xo) ** 2 + (y - yo) ** 2 <= ro ** 2
        inds = on_star & under_occultor
        g /= np.count_nonzero(under_occultor)

        # The solution vector for the complement of the visible region
        sTbar = np.pi * ro ** 2 * np.sum(g[:, inds], axis=1)

        # The solution vector over the entire star
        sT0 = b * self.sT0

        return sT0 - sTbar


@pytest.fixture(scope="module")
def solvers(lmax=5):
    return StarrySolver(lmax), BruteSolver(lmax)


class Compare:
    """
    Base class for comparing the `starry` solver to the brute solver.

    """

    def test_compare(self, solvers, kwargs):
        atol = kwargs.get("atol", 2.0e-6)
        sT_starry = solvers[0].get_sT(**kwargs)
        sT_brute = solvers[1].get_sT(**kwargs)
        diff = np.abs(sT_starry - sT_brute)
        print(
            "Med: {:.3e} | Max: {:.3e} | Tol: {:.3e}".format(
                np.median(diff), np.max(diff), atol
            )
        )
        if kwargs.get("complete", False):
            assert np.allclose(sT_starry, 0.0, atol)
        else:
            assert np.allclose(sT_starry, sT_brute, atol=atol)


@pytest.mark.parametrize("kwargs", [dict(bo=0.5, ro=0.1, f=0.1, theta=0.5)])
class TestOccultorDoesntTouchLimb(Compare):
    pass


@pytest.mark.parametrize("kwargs", [dict(bo=0.85, ro=0.1, f=0.1, theta=0.5)])
class TestKsqGreaterThanOne(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.91, ro=0.1, f=0.1, theta=0.8),
        dict(bo=0.85, ro=0.1, f=0.1, theta=0.1),
    ],
)
class TestKsqLessThanOne(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.95, ro=0.1, f=0.0, theta=0.8),
        dict(bo=0.95, ro=0.1, f=1e-15, theta=0.8),
        dict(bo=0.95, ro=0.1, f=1e-12, theta=0.8),
    ],
)
class TestNearlySpherical(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.0, ro=0.1, f=0.2, theta=0.8),
        dict(bo=1e-15, ro=0.1, f=0.1, theta=0.8),
        dict(bo=1e-12, ro=0.1, f=0.1, theta=0.8),
    ],
)
class TestSmallImpactParameter(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.5, ro=1e-6, f=0.1, theta=0.5),
        dict(bo=0.5, ro=1e-4, f=0.1, theta=0.5),
    ],
)
class TestSmallRadius(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.85, ro=0.1, f=0.1, theta=0.0),
        dict(bo=0.85, ro=0.1, f=0.1, theta=0.5 * np.pi),
    ],
)
class TestSinOrCosThetaIsZero(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.1, ro=0.1, f=0.1, theta=0.5),
        dict(bo=0.5, ro=0.5, f=0.1, theta=0.5, atol=1e-4),
    ],
)
class TestBoEqualsRo(Compare):
    pass


@pytest.mark.parametrize("kwargs", [dict(bo=0.9, ro=0.1, f=0.1, theta=0.5)])
class TestBoEqualsOneMinusRo(Compare):
    pass


@pytest.mark.parametrize("kwargs", [dict(bo=2.0, ro=0.1, f=0.1, theta=0.5)])
class TestNoOccultation(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs", [dict(bo=0.3, ro=2.0, f=0.1, theta=0.5, complete=True)]
)
class TestCompleteOccultation(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs", [dict(bo=1.00058, ro=0.1, f=0.1, theta=-1.26629)]
)
class TestKsqLessThanHalf(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=1.0998151795596858, ro=0.1, f=0.1, theta=-1.4797462082413333),
        dict(bo=1.0476234129628497, ro=0.05, f=0.1, theta=-1.28036820589747),
        dict(
            bo=1.004987562112089,
            ro=0.1,
            f=0.1,
            theta=1.4711276743037345,
            atol=1e-5,
        ),
        dict(bo=1.0956051937836104, ro=0.1, f=0.1, theta=1.5616688245611983),
        dict(
            bo=1.4251233204133293,
            ro=0.5,
            f=0.1,
            theta=1.3587016514315267,
            atol=1e-4,
        ),
        dict(bo=0.9568856885688621, ro=0.1, f=0.25, theta=-1.5707962222892058),
        dict(bo=1.0961596159615965, ro=0.1, f=0.1, theta=0.5 * np.pi),
    ],
)
class TestEdgeCases(Compare):
    pass
