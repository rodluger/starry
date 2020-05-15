# -*- coding: utf-8 -*-
"""
Linear solve / likelihood tests.

"""
import starry
import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import multivariate_normal
import pytest
import itertools

# Parameter combinations we'll test
vals = ["scalar", "vector", "matrix", "cholesky"]
woodbury = [False, True]
solve_inputs = itertools.product(vals, vals)
lnlike_inputs = itertools.product(vals, vals, woodbury)

# Instantiate a dipole map
map = starry.Map(ydeg=1, reflected=True)
amp_true = 0.75
inc_true = 60
y_true = np.array([1, 0.1, 0.2, 0.3])
map.amp = amp_true
map[1, :] = y_true[1:]
map.inc = inc_true

# Generate a synthetic light curve with just a little noise
theta = np.linspace(0, 360, 100)
phi = 3.5 * theta
xs = np.cos(phi * np.pi / 180)
ys = 0.1 * np.cos(phi * np.pi / 180)
zs = np.sin(phi * np.pi / 180)
kwargs = dict(theta=theta, xs=xs, ys=ys, zs=zs)
flux = map.flux(**kwargs).eval()
sigma = 1e-5
np.random.seed(1)
flux += np.random.randn(len(theta)) * sigma


@pytest.mark.parametrize("L,C", solve_inputs)
def test_solve(L, C):
    # Place a generous prior on the map coefficients
    if L == "scalar":
        map.set_prior(L=1)
    elif L == "vector":
        map.set_prior(L=np.ones(map.Ny))
    elif L == "matrix":
        map.set_prior(L=np.eye(map.Ny))
    elif L == "cholesky":
        map.set_prior(cho_L=np.eye(map.Ny))

    # Provide the dataset
    if C == "scalar":
        map.set_data(flux, C=sigma ** 2)
    elif C == "vector":
        map.set_data(flux, C=np.ones(len(flux)) * sigma ** 2)
    elif C == "matrix":
        map.set_data(flux, C=np.eye(len(flux)) * sigma ** 2)
    elif C == "cholesky":
        map.set_data(flux, cho_C=np.eye(len(flux)) * sigma)

    # Solve the linear problem
    map.inc = inc_true
    mu, cho_cov = map.solve(**kwargs)
    mu = mu.eval()
    cho_cov = cho_cov.eval()

    # Ensure the likelihood of the true value is close to that of
    # the MAP solution
    cov = cho_cov.dot(cho_cov.T)
    LnL0 = multivariate_normal.logpdf(mu, mean=mu, cov=cov)
    LnL = multivariate_normal.logpdf(amp_true * y_true, mean=mu, cov=cov)
    assert LnL0 - LnL < 5.00

    # Check that we can draw from the posterior
    map.draw()


@pytest.mark.parametrize("L,C,woodbury", lnlike_inputs)
def test_lnlike(L, C, woodbury):
    """Test the log marginal likelihood method."""
    # Place a generous prior on the map coefficients
    if L == "scalar":
        map.set_prior(L=1)
    elif L == "vector":
        map.set_prior(L=np.ones(map.Ny))
    elif L == "matrix":
        map.set_prior(L=np.eye(map.Ny))
    elif L == "cholesky":
        map.set_prior(cho_L=np.eye(map.Ny))

    # Provide the dataset
    if C == "scalar":
        map.set_data(flux, C=sigma ** 2)
    elif C == "vector":
        map.set_data(flux, C=np.ones(len(flux)) * sigma ** 2)
    elif C == "matrix":
        map.set_data(flux, C=np.eye(len(flux)) * sigma ** 2)
    elif C == "cholesky":
        map.set_data(flux, cho_C=np.eye(len(flux)) * sigma)

    # Compute the marginal log likelihood for different inclinations
    incs = [15, 30, 45, 60, 75, 90]
    ll = np.zeros_like(incs, dtype=float)
    for i, inc in enumerate(incs):
        map.inc = inc
        ll[i] = map.lnlike(woodbury=woodbury, **kwargs).eval()

    # Verify that we get the correct inclination
    assert incs[np.argmax(ll)] == 60
    assert np.allclose(ll[np.argmax(ll)], 974.221605)  # benchmarked
