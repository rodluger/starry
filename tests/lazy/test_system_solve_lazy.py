# -*- coding: utf-8 -*-
"""
System linear solve tests.

"""
import starry
import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import multivariate_normal


def test_solve():
    # Instantiate a star with a dipole map
    A = starry.Primary(starry.Map(ydeg=1), prot=0.0)
    y_true = [0.1, 0.2, 0.3]
    A.map[1, :] = y_true

    # Instantiate two transiting planets with different longitudes of
    # ascending node. This ensures there's no null space!
    b = starry.Secondary(starry.Map(), porb=1.0, r=0.1, t0=-0.05, Omega=30.0)
    c = starry.Secondary(starry.Map(), porb=1.0, r=0.1, t0=0.05, Omega=-30.0)
    sys = starry.System(A, b, c)

    # Generate a synthetic light curve with just a little noise
    t = np.linspace(-0.1, 0.1, 100)
    flux = sys.flux(t).eval()
    sigma = 1e-5
    np.random.seed(0)
    flux += np.random.randn(len(t)) * sigma

    # Place a generous prior on the map coefficients
    A.map.set_prior(L=1)

    # Specify the covariance in every way possible: as a scalar,
    # as a vector, as a matrix, or as a Cholesky factorization
    for C, cho_C in zip(
        [
            sigma ** 2,
            sigma ** 2 * np.ones_like(flux),
            sigma ** 2 * np.eye(len(flux)),
            None,
        ],
        [None, None, None, sigma * np.eye(len(flux))],
    ):
        sys.set_data(flux, C=C, cho_C=cho_C)

        # Solve the linear problem
        mu, cho_cov = sys.solve(t=t)

        # Ensure the likelihood of the true value is close to that of
        # the MAP solution
        mean = mu[0].eval()
        cov = cho_cov[0].eval().dot(cho_cov[0].eval().T)
        LnL0 = multivariate_normal.logpdf(mean, mean=mean, cov=cov)
        LnL = multivariate_normal.logpdf([0.1, 0.2, 0.3], mean=mean, cov=cov)
        assert LnL0 - LnL < 5.00
