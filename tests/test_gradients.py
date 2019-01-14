import starry2
import numpy as np
np.random.seed(44)
formatter = {'float_kind':lambda x: "%.6f" % x}

def compare(kind="default", ydeg=2, udeg=4, nw=2, nt=2, eps=1.e-8, 
            t=0.5, theta=10.0, xo=0.3, yo=0.5, ro=0.1):
    """Test the analytic gradients."""
    # Settings
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
    else:
        raise ValueError("Invalid `kind`.")
    coeffs['y'] = np.concatenate((np.random.randn(*sy1), np.zeros(sy2)))
    coeffs['u'] = np.concatenate((np.random.randn(*su1), np.zeros(su2)))
    params = {}
    if kind == "temporal":
        params['t'] = t
    params['theta'] = theta
    params['xo'] = theta
    params['yo'] = yo
    params['ro'] = ro

    # Instantiate
    if kind == "default":
        map = starry2.Map(lmax)
    elif kind == "spectral":
        map = starry2.Map(lmax, nw=nw)
    elif kind == "temporal":
        map = starry2.Map(lmax, nt=nt)
    map[:, :] = np.array(coeffs['y'])
    map[:] = np.array(coeffs['u'])

    # Compute the gradient analytically
    _, grad = map.flux(**params, gradient=True)
    grad['y'] = grad['y'][:Ny]
    grad['u'] = grad['u'][:udeg]

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
        if not np.allclose(grad[key], grad_num[key], atol=1e-6):
            print("Mismatch in %s: \nexpected \n%s, \ngot \n%s" % (
                    key, 
                    np.array2string(
                        np.atleast_1d(grad_num[key]), formatter=formatter
                    ), 
                    np.array2string(
                        np.atleast_1d(grad[key]), formatter=formatter
                    )
                )
            )
        # raise AssertionError()

#compare("default")
#compare("spectral")
compare("temporal")