"""
Test gradient computation.

"""
import starry
import numpy as np
import pytest
import itertools
np.random.seed(44)
debug = False


def assert_allclose(name, expected, got, fmt="%.12f", atol=1e-6, rtol=1e-5):
    """Raise an assertion error if two arrays differ."""
    expected = np.atleast_1d(expected)
    got = np.atleast_1d(got)
    if got.size == 0 and expected.size == 0:
        return
    if not np.allclose(expected, got, atol=1e-6):
        formatter = {'float_kind': lambda x: fmt % x}
        msg = "Mismatch in %s: \nexpected \n%s, \ngot \n%s" % (
                name, 
                np.array2string(expected, formatter=formatter), 
                np.array2string(got, formatter=formatter)
            )
        if debug:
            print(msg)
        else:
            raise AssertionError(msg)


@pytest.fixture(
    scope="class",
    params=itertools.product(
        (2,),                   # ydeg
        (0, 2),                 # udeg
        (False,),               # reflected
        (1, 2),                 # nw
        (1, 2),                 # nt
        (False,),               # multi
        (0.75,),                # t
        (0.0, 15.0,),           # theta
        (0.0, 0.3),             # xo
        (0.5,),                 # yo
        (0., 0.1),              # ro
        (87.5, 90.0),           # inc
        (0.0, 30.0,),           # obl
        (1.e-8,)                # eps
    )
)
def settings(request):
    ydeg, udeg, reflected, nw, nt, multi, \
        t, theta, xo, yo, ro, inc, obl, eps = request.param
    if nw > 1 and nt > 1:
        return None
    elif nw > 1:
        map = starry.Map(ydeg=ydeg, udeg=udeg, nw=nw, multi=multi, 
                         reflected=reflected)
    elif nt > 1:
        map = starry.Map(ydeg=ydeg, udeg=udeg, nt=nt, multi=multi, 
                         reflected=reflected)
    else:
        map = starry.Map(ydeg=ydeg, udeg=udeg, multi=multi, 
                         reflected=reflected)
    np.random.seed(41)
    if map._temporal:
        map[1:, :, :] = np.random.randn(nt * ((ydeg + 1) ** 2 - 1))
    elif map._spectral:
        map[1:, :, :] = np.random.randn((ydeg + 1) ** 2 - 1, nw)
    else:
        map[1:, :] = np.random.randn((ydeg + 1) ** 2 - 1)
    if udeg > 0:
        map[1:] = np.random.randn(udeg)
    map.inc = inc
    map.obl = obl
    return map, eps, t, theta, xo, yo, ro


class TestGradients:
    """Test the analytic gradient calculations."""

    def test_gradients(self, settings):
        """Test the analytic gradients."""
        # Get the map & settings
        if settings is None:
            return
        map, eps, t, theta, xo, yo, ro = settings

        # Set params
        coeffs = {}
        if map._temporal or map._spectral:
            coeffs['y'] = map[1:, :, :]
        else:
            coeffs['y'] = map[1:, :]
        if map.udeg > 0:
            coeffs['u'] = map[1:]
        else:
            coeffs['u'] = []
        params = {}
        if map._temporal:
            params['t'] = t
        params['theta'] = theta
        params['xo'] = xo
        params['yo'] = yo
        params['ro'] = ro
        inc = map.inc
        obl = map.obl

        # Compute the gradient analytically
        flux, grad = map.flux(**params, gradient=True)

        # Check that we get the same flux from the non-gradient call
        assert_allclose("flux", map.flux(**params), flux)

        # Compute it numerically
        grad_num = {}
        for key in params.keys():
            param = params[key]
            params[key] = param - eps
            f1 = map.flux(**params)
            params[key] = param + eps
            f2 = map.flux(**params)
            params[key] = param
            grad_num[key] = (f2 - f1) / (2 * eps)
        grad_num['y'] = np.zeros_like(coeffs['y'])
        n = 0
        for l in range(1, map.ydeg + 1):
            for m in range(-l, l + 1):
                if map._temporal:
                    for t in range(map.nt):
                        epsv = np.zeros(map.nt)
                        epsv[t] = eps
                        map[l, m, :] = coeffs['y'][n] - epsv
                        f1 = map.flux(**params)
                        map[l, m, :] = coeffs['y'][n] + epsv
                        f2 = map.flux(**params)
                        map[l, m, :] = coeffs['y'][n]
                        grad_num['y'][n, t] = (f2 - f1) / (2 * eps)
                else:
                    if map._spectral:
                        map[l, m, :] = coeffs['y'][n] - eps
                        f1 = map.flux(**params)
                        map[l, m, :] = coeffs['y'][n] + eps
                        f2 = map.flux(**params)
                        map[l, m, :] = coeffs['y'][n]
                        grad_num['y'][n] = (f2 - f1) / (2 * eps)
                    else:
                        map[l, m] = coeffs['y'][n] - eps
                        f1 = map.flux(**params)
                        map[l, m] = coeffs['y'][n] + eps
                        f2 = map.flux(**params)
                        map[l, m] = coeffs['y'][n]
                        grad_num['y'][n] = (f2 - f1) / (2 * eps)
                n += 1
        if map._temporal:
            grad_num['y'] = grad_num['y'].T.reshape(-1)
        grad_num['u'] = np.zeros((map.udeg, map.nw))
        n = 0
        for l in range(1, map.udeg + 1):
            map[l] = coeffs['u'][n] - eps
            f1 = map.flux(**params)
            map[l] = coeffs['u'][n] + eps
            f2 = map.flux(**params)
            map[l] = coeffs['u'][n]
            grad_num['u'][n] = (f2 - f1) / (2 * eps)
            n += 1
        map.inc = inc - eps
        f1 = map.flux(**params)
        map.inc = inc + eps
        f2 = map.flux(**params)
        map.inc = inc
        grad_num['inc'] = (f2 - f1) / (2 * eps)
        map.obl = obl - eps
        f1 = map.flux(**params)
        map.obl = obl + eps
        f2 = map.flux(**params)
        map.obl = obl
        grad_num['obl'] = (f2 - f1) / (2 * eps)

        # Compare
        for key in grad.keys():
            assert_allclose(key, np.squeeze(grad_num[key]), np.squeeze(grad[key]))