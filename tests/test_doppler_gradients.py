"""
Test doppler gradient computation.

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
        (0, 2),                 # ydeg
        (0, 2),                 # udeg
        (1,),                   # nw (TODO)
        (1,),                   # nt (TODO)
        (False,),               # multi (TODO)
        (0.75,),                # t
        (0.0, 15.0,),           # theta
        (0.0, 0.3),             # xo
        (0.5,),                 # yo
        (0., 0.1),              # ro
        (87.5, 90.0),           # inc
        (0.0, 30.0,),           # obl
        (0.0, 0.5),             # alpha
        (0.0, 1.0),             # veq
        (1.e-8,)                # eps
    )
)
def settings(request):
    ydeg, udeg, nw, nt, multi, \
        t, theta, xo, yo, ro, inc, obl, \
            alpha, veq, eps = request.param 

    # Disallowed combinations
    if nw > 1 and nt > 1:
        return None
    elif ydeg == 0 and nt > 1:
        return None
    elif (ydeg > 0) and (udeg > 0) and nw > 1:
        return None

    # Allowed combinations
    elif nw > 1:
        map = starry.DopplerMap(ydeg=ydeg, udeg=udeg, nw=nw, multi=multi)
    elif nt > 1:
        map = starry.DopplerMap(ydeg=ydeg, udeg=udeg, nt=nt, multi=multi)
    else:
        map = starry.DopplerMap(ydeg=ydeg, udeg=udeg, multi=multi)

    np.random.seed(41)
    if ydeg > 0:
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
    map.alpha = alpha
    map.veq = veq
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
        if map.ydeg > 0:
            if map._temporal or map._spectral:
                coeffs['y'] = map[1:, :, :]
            else:
                coeffs['y'] = map[1:, :]
        else:
            coeffs['y'] = []
        if map.udeg > 0:
            if map._limbdarkened and map._spectral:
                coeffs['u'] = map[1:, :]
            else:
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
        alpha = map.alpha
        veq = map.veq

        # Compute the gradient analytically
        rv, grad = map.rv(**params, gradient=True)
        rv = np.array(rv)
        for key in grad.keys():
            grad[key] = np.array(grad[key])

        # Check that we get the same rv from the non-gradient call
        assert_allclose("rv", map.rv(**params), rv)

        # Compute it numerically
        grad_num = {}
        for key in params.keys():
            param = params[key]
            params[key] = param - eps
            f1 = np.array(map.rv(**params))
            params[key] = param + eps
            f2 = np.array(map.rv(**params))
            params[key] = param
            grad_num[key] = (f2 - f1) / (2 * eps)
        
        # Spherical harmonic coeffs
        grad_num['y'] = np.zeros_like(coeffs['y'])
        n = 0
        for l in range(1, map.ydeg + 1):
            for m in range(-l, l + 1):
                if map._temporal:
                    for t in range(map.nt):
                        epsv = np.zeros(map.nt)
                        epsv[t] = eps
                        map[l, m, :] = coeffs['y'][n] - epsv
                        f1 = np.array(map.rv(**params))
                        map[l, m, :] = coeffs['y'][n] + epsv
                        f2 = np.array(map.rv(**params))
                        map[l, m, :] = coeffs['y'][n]
                        grad_num['y'][n, t] = (f2 - f1) / (2 * eps)
                else:
                    if map._spectral:
                        map[l, m, :] = coeffs['y'][n] - eps
                        f1 = np.array(map.rv(**params))
                        map[l, m, :] = coeffs['y'][n] + eps
                        f2 = np.array(map.rv(**params))
                        map[l, m, :] = coeffs['y'][n]
                        grad_num['y'][n] = (f2 - f1) / (2 * eps)
                    else:
                        map[l, m] = coeffs['y'][n] - eps
                        f1 = np.array(map.rv(**params))
                        map[l, m] = coeffs['y'][n] + eps
                        f2 = np.array(map.rv(**params))
                        map[l, m] = coeffs['y'][n]
                        grad_num['y'][n] = (f2 - f1) / (2 * eps)
                n += 1
        if map._temporal:
            grad_num['y'] = grad_num['y'].T.reshape(-1)
        
        # Limb darkening coeffs
        grad_num['u'] = np.zeros(map.udeg)
        n = 0
        for l in range(1, map.udeg + 1):
            map[l] = coeffs['u'][n] - eps
            f1 = np.array(map.rv(**params))
            map[l] = coeffs['u'][n] + eps
            f2 = np.array(map.rv(**params))
            map[l] = coeffs['u'][n]
            grad_num['u'][n] = (f2 - f1) / (2 * eps)
            n += 1
        map.inc = inc - eps
        f1 = np.array(map.rv(**params))
        map.inc = inc + eps
        f2 = np.array(map.rv(**params))
        map.inc = inc
        grad_num['inc'] = (f2 - f1) / (2 * eps)
        map.obl = obl - eps
        f1 = np.array(map.rv(**params))
        map.obl = obl + eps
        f2 = np.array(map.rv(**params))
        map.obl = obl
        grad_num['obl'] = (f2 - f1) / (2 * eps)

        # One-sided derivs, since alpha > 0
        map.alpha = alpha
        f1 = np.array(map.rv(**params))
        map.alpha = alpha + eps
        f2 = np.array(map.rv(**params))
        map.alpha = alpha
        grad_num['alpha'] = (f2 - f1) / eps

        # One-sided derivs, since veq > 0
        map.veq = veq
        f1 = np.array(map.rv(**params))
        map.veq = veq + eps
        f2 = np.array(map.rv(**params))
        map.veq = veq
        grad_num['veq'] = (f2 - f1) / eps

        # Compare
        for key in grad.keys():
            #print(key, np.squeeze(grad_num[key]), np.squeeze(grad[key]))
            assert_allclose(key, np.squeeze(grad_num[key]), np.squeeze(grad[key]))