"""Test autograd minimization."""
import starry
import numpy as np
from scipy.optimize import minimize


def assert_allclose(x, y):
    """Shortcut assert with ppm tolerance."""
    return np.testing.assert_allclose(x, y, rtol=1e-6)


def chisq_grad(y, map, xo, yo, ro, theta, flux, sigma=0.001):
    """Return the chisq and its derivative."""
    # Assign the map coefficients
    n = 0
    for l in range(1, map.lmax + 1):
        for m in range(-l, l + 1):
            map[l, m] = y[n]
            n += 1

    # Compute the model and the chi-squared
    model = map.flux(xo=xo, yo=yo, ro=ro, theta=theta)
    chi2 = np.sum((model - flux) ** 2) / sigma ** 2

    # Get the derivatives of the model w/ respect to y
    dmdy = [map.gradient['Y_{%d,%d}' % (l, m)]
            for l in range(1, map.lmax + 1) for m in range(-l, l + 1)]

    # Now compute the gradient of chi-squared with respect to y
    grad = np.sum(2 * (model - flux) * dmdy, axis=1) / sigma ** 2

    # Return chi squared **and** gradient
    return chi2, grad


def test_earth():
    """Test max like on the earth."""
    # Initialize the randomizer
    np.random.seed(8765)

    # Load the Earth map
    lmax = 5
    earth = starry.Map(lmax)
    earth.load_image('earth')

    # Check against benchmarks
    assert_allclose(earth.evaluate(x=[0, 0.2, 0.7, 0.99],
                                   y=[0, 0.3, 0.4, 0]),
                    np.array([0.40127585, 0.66505788,
                              0.41675531, 0.15982961]))
    assert_allclose(np.array([earth[5, m] for m in range(-5, 6)]),
                    np.array([-0.1418789, 0.06168199, -0.19398174,
                              -0.0186274, 0.06740396, -0.02105715,
                              0.06642807, -0.00887614, -0.0654446,
                              0.04427496, -0.00235343]))

    # Generate fake data
    npts = 100
    nevents = 20
    ro = 0.1
    xo = []
    yo = []
    theta = []
    dy = []
    n = 0
    while n < nevents:
        ymin = 2 * np.random.random() - 1
        alpha = np.pi * np.random.random() - np.pi / 2
        x = np.linspace(-1.5, 1.5, npts)
        y = ymin + x * np.tan(alpha)
        inds = x ** 2 + y ** 2 <= 1.5 ** 2
        if len(x[inds]) > 10:
            xo.append(x[inds])
            yo.append(y[inds])
            dy.append(ro / np.cos(alpha))
            theta.append(np.ones_like(x[inds]) * np.random.random() * 360)
            n += 1

    # Check against benchmarks
    flux = [None for i in range(nevents)]
    for i in range(nevents):
        flux[i] = earth.flux(xo=xo[i], yo=yo[i], ro=ro, theta=theta[i])
    assert_allclose(flux[3][10:15], np.array(
        [0.97717466, 0.97717466, 0.97668376, 0.97571509, 0.97447185]))
    assert_allclose(flux[7][30:35], np.array(
        [0.59307336, 0.59276075, 0.59246805, 0.59221149, 0.59200468]))

    # Concatenate all the observations
    xo = np.concatenate(xo)
    yo = np.concatenate(yo)
    theta = np.concatenate(theta)
    flux = np.concatenate(flux)

    # Find the max like solution using gradients
    y = 0.1 * np.random.randn((lmax + 1) ** 2)
    map = starry.grad.Map(lmax)
    map[0, 0] = 1
    res = minimize(chisq_grad, y[1:],
                   args=(map, xo, yo, ro, theta, flux), jac=True)
    assert res.fun < 1e-4


if __name__ == "__main__":
    test_earth()
