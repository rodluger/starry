"""
Test gradient computation.

"""
import starry2
import numpy as np
import pytest
np.random.seed(44)
debug = False


def assert_allclose(name, expected, got, fmt="%.6f", atol=1e-6, rtol=1e-5):
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
        

def compare(ydeg=2, udeg=0, nw=1, nt=1, eps=1.e-8, axis=[0, 1, 0],
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
    lmax = ydeg + udeg
    N = (lmax + 1) ** 2
    Ny = (ydeg + 1) ** 2
    coeffs = {}
    if kind == "default":
        sy1 = (Ny,)
        sy2 = (N - Ny,)
        su1 = (udeg,)
        su2 = (lmax - udeg,)
    elif kind == "spectral":
        sy1 = (Ny, nw)
        sy2 = (N - Ny, nw)
        su1 = (udeg, nw)
        su2 = (lmax - udeg, nw)
    elif kind == "temporal":
        sy1 = (Ny, nt)
        sy2 = (N - Ny, nt)
        su1 = (udeg,)
        su2 = (lmax - udeg,)
    coeffs['y'] = np.concatenate((np.random.randn(*sy1), np.zeros(sy2)))
    coeffs['u'] = np.concatenate((np.random.randn(*su1), np.zeros(su2)))
    params = {}
    if kind == "temporal":
        params['t'] = t
    params['theta'] = theta
    params['xo'] = xo
    params['yo'] = yo
    params['ro'] = ro

    # Instantiate
    if kind == "default":
        map = starry2.Map(lmax, multi=multi)
    elif kind == "spectral":
        map = starry2.Map(lmax, nw=nw, multi=multi)
    elif kind == "temporal":
        map = starry2.Map(lmax, nt=nt, multi=multi)
    map.axis = axis
    map[:, :] = np.array(coeffs['y'])
    map[:] = np.array(coeffs['u'])

    # Compute the gradient analytically
    flux, grad = map.flux(**params, gradient=True)
    grad['y'] = grad['y'][:Ny]
    grad['u'] = grad['u'][:udeg]

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
    grad_num['y'] = np.zeros(sy1)
    n = 0
    for l in range(ydeg + 1):
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
    grad_num['u'] = np.zeros(su1)
    n = 0
    for l in range(1, udeg + 1):
        map[l] = coeffs['u'][n] - eps
        f1 = map.flux(**params)
        map[l] = coeffs['u'][n] + eps
        f2 = map.flux(**params)
        map[l] = coeffs['u'][n]
        grad_num['u'][n] = (f2 - f1) / (2 * eps)
        n += 1

    # Compare
    for key in grad.keys():
        assert_allclose(key, grad_num[key], grad[key])


def test_default_ylm_phase():
    compare(ydeg=2, udeg=0, nw=1, nt=1, theta=15.0, xo=10.0)


def test_default_ylm_occ():
    compare(ydeg=2, udeg=0, nw=1, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_default_ld_phase():
    compare(ydeg=0, udeg=2, nw=1, nt=1, theta=15.0, xo=10.0)


def test_default_ld_occ():
    compare(ydeg=0, udeg=2, nw=1, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_default_ylmld_phase():
    compare(ydeg=2, udeg=2, nw=1, nt=1, theta=15.0, xo=10.0)


def test_default_ylmld_occ():
    compare(ydeg=2, udeg=2, nw=1, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_spectral_ylm_phase():
    compare(ydeg=2, udeg=0, nw=3, nt=1, theta=15.0, xo=10.0)


def test_spectral_ylm_occ():
    compare(ydeg=2, udeg=0, nw=3, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_spectral_ld_phase():
    compare(ydeg=0, udeg=2, nw=3, nt=1, theta=15.0, xo=10.0)


def test_spectral_ld_occ():
    compare(ydeg=0, udeg=2, nw=3, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_spectral_ylmld_phase():
    compare(ydeg=2, udeg=2, nw=3, nt=1, theta=15.0, xo=10.0)


def test_spectral_ylmld_occ():
    compare(ydeg=2, udeg=2, nw=3, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_temporal_ylm_phase():
    compare(ydeg=2, udeg=0, nw=1, nt=3, t=0.75, theta=15.0, xo=10.0)


def test_temporal_ylm_occ():
    compare(ydeg=2, udeg=0, nw=1, nt=3, t=0.75, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_temporal_ld_phase():
    compare(ydeg=0, udeg=2, nw=1, nt=3, t=0.75, theta=15.0, xo=10.0)


def test_temporal_ld_occ():
    compare(ydeg=0, udeg=2, nw=1, nt=3, t=0.75, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_temporal_ylmld_phase():
    compare(ydeg=2, udeg=2, nw=1, nt=3, t=0.75, theta=15.0, xo=10.0)


def test_temporal_ylmld_occ():
    compare(ydeg=2, udeg=2, nw=1, nt=3, t=0.75, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_default_ylm_phase_multi():
    compare(multi=True, ydeg=2, udeg=0, nw=1, nt=1, theta=15.0, xo=10.0)


def test_default_ylm_occ_multi():
    compare(multi=True, ydeg=2, udeg=0, nw=1, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_default_ld_phase_multi():
    compare(multi=True, ydeg=0, udeg=2, nw=1, nt=1, theta=15.0, xo=10.0)


def test_default_ld_occ_multi():
    compare(multi=True, ydeg=0, udeg=2, nw=1, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_default_ylmld_phase_multi():
    compare(multi=True, ydeg=2, udeg=2, nw=1, nt=1, theta=15.0, xo=10.0)


def test_default_ylmld_occ_multi():
    compare(multi=True, ydeg=2, udeg=2, nw=1, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_spectral_ylm_phase_multi():
    compare(multi=True, ydeg=2, udeg=0, nw=3, nt=1, theta=15.0, xo=10.0)


def test_spectral_ylm_occ_multi():
    compare(multi=True, ydeg=2, udeg=0, nw=3, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_spectral_ld_phase_multi():
    compare(multi=True, ydeg=0, udeg=2, nw=3, nt=1, theta=15.0, xo=10.0)


def test_spectral_ld_occ_multi():
    compare(multi=True, ydeg=0, udeg=2, nw=3, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_spectral_ylmld_phase_multi():
    compare(multi=True, ydeg=2, udeg=2, nw=3, nt=1, theta=15.0, xo=10.0)


def test_spectral_ylmld_occ_multi():
    compare(multi=True, ydeg=2, udeg=2, nw=3, nt=1, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_temporal_ylm_phase_multi():
    compare(multi=True, ydeg=2, udeg=0, nw=1, nt=3, t=0.75, theta=15.0, xo=10.0)


def test_temporal_ylm_occ_multi():
    compare(multi=True, ydeg=2, udeg=0, nw=1, nt=3, t=0.75, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_temporal_ld_phase_multi():
    compare(multi=True, ydeg=0, udeg=2, nw=1, nt=3, t=0.75, theta=15.0, xo=10.0)


def test_temporal_ld_occ_multi():
    compare(multi=True, ydeg=0, udeg=2, nw=1, nt=3, t=0.75, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


def test_temporal_ylmld_phase_multi():
    compare(multi=True, ydeg=2, udeg=2, nw=1, nt=3, t=0.75, theta=15.0, xo=10.0)


def test_temporal_ylmld_occ_multi():
    compare(multi=True, ydeg=2, udeg=2, nw=1, nt=3, t=0.75, theta=15.0, xo=0.3, yo=0.5, ro=0.1)


if __name__ == "__main__":
    debug = True
    test_default_ylmld_occ_multi()