# -*- coding: utf-8 -*-
from .. import config
from .utils import *
import theano
import theano.tensor as tt
import theano.tensor.slinalg as sla
import numpy as np
import scipy


__all__ = ["linalg", "Covariance"]


class Covariance(object):
    """A container for covariance matrices."""

    def __init__(self, C=None, cho_C=None, N=None):

        # User provided the Cholesky factorization
        if cho_C is not None:

            self.cholesky = math.cast(cho_C)
            self.matrix = math.dot(
                self.cholesky, math.transpose(self.cholesky)
            )
            self.inverse = self._cho_solve(self.cholesky)
            self.lndet = 2 * math.sum(math.log(math.diag(self.cholesky)))
            self.kind = "cholesky"

        # User provided the covariance as a scalar, vector, or matrix
        elif C is not None:

            C = math.cast(C)

            if hasattr(C, "ndim"):

                if C.ndim == 0:

                    assert N is not None, "Please provide a matrix size `N`."
                    self.cholesky = math.sqrt(C) * math.eye(N)
                    self.inverse = (1.0 / C) * math.eye(N)
                    self.lndet = N * math.log(C)
                    self.matrix = C * math.eye(N)
                    self.kind = "scalar"

                elif C.ndim == 1:

                    self.cholesky = math.diag(math.sqrt(C))
                    self.inverse = math.diag(1.0 / C)
                    self.lndet = math.sum(math.log(C))
                    self.matrix = math.diag(C)
                    self.kind = "vector"

                else:

                    self.cholesky = math.cholesky(C)
                    self.inverse = self._cho_solve(self.cholesky)
                    self.lndet = 2 * math.sum(
                        math.log(math.diag(self.cholesky))
                    )
                    self.matrix = C
                    self.kind = "matrix"

            # Assume it's a scalar
            else:

                assert N is not None, "Please provide a matrix size `N`."
                self.cholesky = math.sqrt(C) * math.eye(N)
                self.inverse = (1.0 / C) * math.eye(N)
                self.lndet = N * math.log(C)
                self.matrix = C * math.eye(N)
                self.kind = "scalar"

        # ?!
        else:
            raise ValueError("Either `C` or `cho_C` must be provided.")

    def _cho_solve(self, cho_A, b=None):
        """Apply the cholesky factorization to a vector or matrix."""
        if config.lazy:
            if b is None:
                b = tt.eye(cho_A.shape[0])
            solve_lower = sla.Solve(A_structure="lower_triangular", lower=True)
            solve_upper = sla.Solve(
                A_structure="upper_triangular", lower=False
            )
            return solve_upper(tt.transpose(cho_A), solve_lower(cho_A, b))
        else:
            if b is None:
                b = np.eye(cho_A.shape[0])
            return scipy.linalg.cho_solve((cho_A, True), b)


class OpsLinAlg(object):
    """Linear algebra operations for maps."""

    def __init__(self):
        solve_lower = sla.Solve(A_structure="lower_triangular", lower=True)
        solve_upper = sla.Solve(A_structure="upper_triangular", lower=False)
        self.cho_solve = lambda cho_A, b: solve_upper(
            tt.transpose(cho_A), solve_lower(cho_A, b)
        )

    @autocompile(
        "MAP",
        tt.dmatrix(),
        tt.dvector(),
        tt.dmatrix(),
        tt.dvector(),
        tt.dmatrix(),
    )
    def MAP(self, X, flux, cho_C, mu, cho_L):
        """
        Compute the maximum a posteriori (MAP) prediction for the
        spherical harmonic coefficients of a map given a flux timeseries.

        Args:
            X: The flux design matrix.
            flux (ndarray): The flux timeseries.
            cho_C: The lower cholesky factorization of the data covariance.
            mu: The prior mean of the spherical harmonic coefficients.
            cho_L: The lower cholesky factorization of the prior covariance of the
                spherical harmonic coefficients.

        Returns:
            The vector of spherical harmonic coefficients corresponding to the
            MAP solution and the Cholesky factorization of the corresponding
            covariance matrix.

        """
        # Compute C^-1 . X
        CInvX = self.cho_solve(cho_C, X)

        # Compute W = X^T . C^-1 . X + L^-1
        W = tt.dot(tt.transpose(X), CInvX) + self.cho_solve(
            cho_L, tt.eye(X.shape[1])
        )

        # Compute the max like y and its covariance matrix
        cho_W = sla.cholesky(W)
        M = self.cho_solve(cho_W, tt.transpose(CInvX))
        yhat = tt.dot(M, flux) + self.cho_solve(cho_L, mu)
        ycov = self.cho_solve(cho_W, tt.eye(cho_W.shape[0]))
        cho_ycov = sla.cholesky(ycov)

        return yhat, cho_ycov

    @autocompile(
        "lnlike",
        tt.dmatrix(),
        tt.dvector(),
        tt.dmatrix(),
        tt.dvector(),
        tt.dmatrix(),
    )
    def lnlike(self, X, flux, C, mu, L):
        """
        Compute the log marginal likelihood of the data given a design matrix.

        Args:
            X: The flux design matrix.
            flux (ndarray): The flux timeseries.
            C: The data covariance matrix.
            mu: The prior mean of the spherical harmonic coefficients.
            L: The prior covariance of the spherical harmonic coefficients.

        Returns:
            The log marginal likelihood of the `flux` vector conditioned on
            the design matrix `X`. This is the likelihood marginalized over
            all possible spherical harmonic vectors, which is analytically
            computable for the linear `starry` model.

        """

        # Compute the GP
        gp_mu = tt.dot(X, mu)
        gp_cov = C + tt.dot(tt.dot(X, L), tt.transpose(X))
        cho_gp_cov = sla.cholesky(gp_cov)

        # Compute the marginal likelihood
        N = X.shape[0]
        r = tt.reshape(flux - gp_mu, (-1, 1))
        lnlike = -0.5 * tt.dot(tt.transpose(r), self.cho_solve(cho_gp_cov, r))
        lnlike -= tt.sum(tt.log(tt.diag(cho_gp_cov)))
        lnlike -= 0.5 * N * tt.log(2 * np.pi)

        return lnlike[0, 0]


# Instantiate the Op
linalg = OpsLinAlg()
