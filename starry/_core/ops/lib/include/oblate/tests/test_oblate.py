import oblate
import numpy as np
import pytest


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
        ids=["cpp-py", "cpp-lin", "cpp-num", "cpp-brute", "brute-num"],
    )
    def test_compare(self, solvers, kwargs, compare, request):
        kwargs = dict(kwargs)
        max_error_cpp_py = kwargs.pop("max_error_cpp_py", 1e-12)
        max_error_cpp_lin = kwargs.pop("max_error_cpp_lin", 1e-12)
        max_error_cpp_num = kwargs.pop("max_error_cpp_num", 2e-3)
        max_error_cpp_brute = kwargs.pop("max_error_cpp_brute", 2e-3)
        max_error_brute_num = kwargs.pop("max_error_brute_num", 2e-3)
        atol = {
            ("cpp", "py"): max_error_cpp_py,
            ("cpp", "lin"): max_error_cpp_lin,
            ("cpp", "num"): max_error_cpp_num,
            ("cpp", "brute"): max_error_cpp_brute,
            ("brute", "num"): max_error_brute_num,
        }
        try:
            sT0 = solvers[compare[0]].get_sT(**kwargs)
            sT1 = solvers[compare[1]].get_sT(**kwargs)
            diff = np.abs(sT0 - sT1)
            print(
                "Med: {:.3e} | Max: {:.3e} | Tol: {:.3e}".format(
                    np.median(diff), np.max(diff), atol[compare]
                )
            )
            assert np.allclose(sT0, sT1, atol=atol[compare])
        except Exception as e:
            file = "{}.pdf".format(request.node.location[-1])
            oblate.draw(file=file, **kwargs)
            raise e


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
        dict(bo=0.95, ro=0.1, f=0.0, theta=0.8, max_error_cpp_num=1e-12),
        dict(bo=0.95, ro=0.1, f=1e-15, theta=0.8, max_error_cpp_num=1e-12),
        dict(bo=0.95, ro=0.1, f=1e-12, theta=0.8, max_error_cpp_num=1e-12),
    ],
)
class TestNearlySpherical(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.0, ro=0.1, f=0.2, theta=0.8, max_error_cpp_num=5e-2),
        dict(bo=1e-15, ro=0.1, f=0.1, theta=0.8),
        dict(bo=1e-12, ro=0.1, f=0.1, theta=0.8),
    ],
)
class TestSmallImpactParameter(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bo=0.5, ro=1e-6, f=0.1, theta=0.5, max_error_cpp_num=1e-12),
        dict(bo=0.5, ro=1e-4, f=0.1, theta=0.5, max_error_cpp_num=1e-9),
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
        dict(
            bo=0.5,
            ro=0.5,
            f=0.1,
            theta=0.5,
            max_error_cpp_lin=1e-5,
            max_error_cpp_num=5e-3,
            max_error_cpp_brute=5e-3,
        ),
    ],
)
class TestBoEqualsRo(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [dict(bo=0.9, ro=0.1, f=0.1, theta=0.5, max_error_cpp_lin=1e-6),],
)
class TestBoEqualsOneMinusRo(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs", [dict(bo=2.0, ro=0.1, f=0.1, theta=0.5),],
)
class TestNoOccultation(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs", [dict(bo=0.3, ro=2.0, f=0.1, theta=0.5),],
)
class TestCompleteOccultation(Compare):
    pass


@pytest.mark.parametrize(
    "kwargs", [dict(bo=1.00058, ro=0.1, f=0.1, theta=-1.26629)],
)
class TestKsqLessThanHalf(Compare):
    """
    These configurations correspond to

        1 - bo ** 2 - ro ** 2 < 0
    
    which is a special case for the J integral.
    """

    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            bo=1.0998151795596858, ro=0.1, f=0.1, theta=-1.4797462082413333
        ),  # This shouldn't be an occultation!
        dict(
            bo=1.004987562112089, ro=0.1, f=0.1, theta=1.4711276743037345
        ),  # Angle finding failed?
        dict(
            bo=1.0961596159615965, ro=0.1, f=0.1, theta=1.5707963267948966
        ),  # P integration has the wrong sign
        dict(
            bo=1.0956051937836104, ro=0.1, f=0.1, theta=1.5616688245611983
        ),  # P integration has the wrong sign
        dict(
            bo=1.0476234129628497, ro=0.05, f=0.1, theta=-1.28036820589747
        ),  # This shouldn't be an occultation!
        dict(
            bo=1.4251233204133293, ro=0.5, f=0.1, theta=1.3587016514315267
        ),  # P integration has the wrong sign
    ],
)
class TestFailing(Compare):
    """
    Currently failing tests.
    """

    pass
