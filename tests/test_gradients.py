"""
Test gradient computation.

"""
import starry
import numpy as np
import pytest
np.random.seed(44)
debug = True


def assert_allclose(name, expected, got, fmt="%.12f", atol=1e-6, rtol=1e-5):
    """Raise an assertion error if two arrays differ."""
    expected = np.atleast_1d(expected)
    got = np.atleast_1d(got)
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
        

def compare(ydeg=2, udeg=2, nw=1, nt=1, reflected=False, eps=1.e-8, inc=87.5, obl=30,
            multi=False, t=0.75, theta=15.0, xo=0.3, yo=0.5, ro=0.1):
    """Test the analytic gradients."""
    # Settings
    if nw > 1:
        kind = "spectral"
    elif nt > 1:
        kind = "temporal"
    elif nw == 1 and nt == 1:
        kind = "default"
    else:
        raise ValueError()

    coeffs = {}
    coeffs['y'] = np.random.randn(nw * nt * ((ydeg + 1) ** 2 - 1))
    coeffs['u'] = np.random.randn(udeg)
    params = {}
    if kind == "temporal":
        params['t'] = t
    params['theta'] = theta
    params['xo'] = xo
    params['yo'] = yo
    params['ro'] = ro

    # Instantiate
    if kind == "default":
        map = starry.Map(ydeg=ydeg, udeg=udeg, multi=multi, reflected=reflected)
    elif kind == "spectral":
        map = starry.Map(ydeg=ydeg, udeg=udeg, nw=nw, multi=multi, reflected=reflected)
    elif kind == "temporal":
        map = starry.Map(ydeg=ydeg, udeg=udeg, nt=nt, multi=multi, reflected=reflected)
    map.inc = inc
    map.obl = obl
    map[1:, :] = np.array(coeffs['y'])
    if udeg > 0:
        map[1:] = np.array(coeffs['u'])

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
    for l in range(1, ydeg + 1):
        for m in range(-l, l + 1):
            if kind == "temporal":
                for t in range(nt):
                    epsv = np.zeros(nt)
                    epsv[t] = eps
                    map[l, m] = coeffs['y'][n] - epsv
                    f1 = map.flux(**params)
                    map[l, m] = coeffs['y'][n] + epsv
                    f2 = map.flux(**params)
                    map[l, m] = coeffs['y'][n]
                    grad_num['y'][n, t] = (f2 - f1) / (2 * eps)
            else:
                map[l, m] = coeffs['y'][n] - eps
                f1 = map.flux(**params)
                map[l, m] = coeffs['y'][n] + eps
                f2 = map.flux(**params)
                map[l, m] = coeffs['y'][n]
                grad_num['y'][n] = (f2 - f1) / (2 * eps)
            n += 1
    grad_num['u'] = np.zeros_like(coeffs['u'])
    n = 0
    for l in range(1, udeg + 1):
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
        assert_allclose(key, grad_num[key], np.squeeze(grad[key]))

# DEBUG
compare(ro=0)