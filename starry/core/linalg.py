# -*- coding: utf-8 -*-
from .. import config
from .utils import *
from .math import math
import theano
import theano.tensor as tt
import theano.tensor.slinalg as sla
import numpy as np
import scipy


__all__ = ["linalg"]


class _Covariance(object):
    """A container for covariance matrices."""

    def __init__(self, C=None, cho_C=None, N=None):

        # User provided the Cholesky factorization
        if cho_C is not None:

            self.cholesky = math.cast(cho_C)
            self.value = math.dot(self.cholesky, math.transpose(self.cholesky))
            self.inverse = self._cho_solve(self.cholesky)
            self.lndet = 2 * math.sum(math.log(math.diag(self.cholesky)))
            self.kind = "cholesky"
            self.N = cho_C.shape[0]

        # User provided the covariance as a scalar, vector, or matrix
        elif C is not None:

            C = math.cast(C)

            if hasattr(C, "ndim"):

                if C.ndim == 0:

                    assert N is not None, "Please provide a matrix size `N`."
                    self.cholesky = math.sqrt(C)
                    self.inverse = math.cast(1.0 / C)
                    self.lndet = math.cast(N * math.log(C))
                    self.value = C
                    self.kind = "scalar"
                    self.N = N

                elif C.ndim == 1:

                    self.cholesky = math.sqrt(C)
                    self.inverse = 1.0 / C
                    self.lndet = math.sum(math.log(C))
                    self.value = C
                    self.kind = "vector"
                    self.N = C.shape[0]

                else:

                    self.cholesky = math.cholesky(C)
                    self.inverse = self._cho_solve(self.cholesky)
                    self.lndet = 2 * math.sum(
                        math.log(math.diag(self.cholesky))
                    )
                    self.value = C
                    self.kind = "matrix"
                    self.N = C.shape[0]

            # Assume it's a scalar
            else:

                assert N is not None, "Please provide a matrix size `N`."
                self.cholesky = math.sqrt(C)
                self.inverse = math.cast(1.0 / C)
                self.lndet = math.cast(N * math.log(C))
                self.value = C
                self.kind = "scalar"
                self.N = N

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
        self._solve_lower = sla.Solve(
            A_structure="lower_triangular", lower=True
        )
        self._solve_upper = sla.Solve(
            A_structure="upper_triangular", lower=False
        )

    @autocompile
    def cho_solve(self, cho_A, b):
        return self._solve_upper(
            tt.transpose(cho_A), self._solve_lower(cho_A, b)
        )

    @autocompile
    def MAP(self, X, flux, cho_C, mu, LInv):
        """
        Compute the maximum a posteriori (MAP) prediction for the
        spherical harmonic coefficients of a map given a flux timeseries.

        Args:
            X (matrix): The flux design matrix.
            flux (array): The flux timeseries.
            cho_C (scalar/vector/matrix): The lower cholesky factorization
                of the data covariance.
            mu (array): The prior mean of the spherical harmonic coefficients.
            LInv (scalar/vector/matrix): The inverse prior covariance of the
                spherical harmonic coefficients.

        Returns:
            The vector of spherical harmonic coefficients corresponding to the
            MAP solution and the Cholesky factorization of the corresponding
            covariance matrix.

        """
        # Compute C^-1 . X
        if cho_C.ndim == 0:
            CInvX = X / cho_C ** 2
        elif cho_C.ndim == 1:
            CInvX = tt.dot(tt.diag(1 / cho_C ** 2), X)
        else:
            CInvX = self.cho_solve(cho_C, X)

        # Compute W = X^T . C^-1 . X + L^-1
        W = tt.dot(tt.transpose(X), CInvX)
        if LInv.ndim == 0:
            W = tt.inc_subtensor(
                W[tuple((tt.arange(W.shape[0]), tt.arange(W.shape[0])))], LInv
            )
            LInvmu = mu * LInv
        elif LInv.ndim == 1:
            W = tt.inc_subtensor(
                W[tuple((tt.arange(W.shape[0]), tt.arange(W.shape[0])))], LInv
            )
            LInvmu = mu * LInv
        else:
            W += LInv
            LInvmu = tt.dot(LInv, mu)

        # Compute the max like y and its covariance matrix
        cho_W = sla.cholesky(W)
        M = self.cho_solve(cho_W, tt.transpose(CInvX))
        yhat = tt.dot(M, flux) + LInvmu
        ycov = self.cho_solve(cho_W, tt.eye(cho_W.shape[0]))
        cho_ycov = sla.cholesky(ycov)

        return yhat, cho_ycov

    @autocompile
    def lnlike(self, X, flux, C, mu, L):
        """
        Compute the log marginal likelihood of the data given a design matrix.

        Args:
            X (matrix): The flux design matrix.
            flux (array): The flux timeseries.
            C (scalar/vector/matrix): The data covariance matrix.
            mu (array): The prior mean of the spherical harmonic coefficients.
            L (scalar/vector/matrix): The prior covariance of the spherical
                harmonic coefficients.

        Returns:
            The log marginal likelihood of the `flux` vector conditioned on
            the design matrix `X`. This is the likelihood marginalized over
            all possible spherical harmonic vectors, which is analytically
            computable for the linear `starry` model.

        """
        # Compute the GP mean
        gp_mu = tt.dot(X, mu)

        # Compute the GP covariance
        if L.ndim == 0:
            XLX = tt.dot(X, tt.transpose(X)) * L
        elif L.ndim == 1:
            XLX = tt.dot(tt.dot(X, tt.diag(L)), tt.transpose(X))
        else:
            XLX = tt.dot(tt.dot(X, L), tt.transpose(X))

        if C.ndim == 0 or C.ndim == 1:
            gp_cov = tt.inc_subtensor(
                XLX[tuple((tt.arange(XLX.shape[0]), tt.arange(XLX.shape[0])))],
                C,
            )
        else:
            gp_cov = C + XLX

        cho_gp_cov = sla.cholesky(gp_cov)

        # Compute the marginal likelihood
        N = X.shape[0]
        r = tt.reshape(flux - gp_mu, (-1, 1))
        lnlike = -0.5 * tt.dot(tt.transpose(r), self.cho_solve(cho_gp_cov, r))
        lnlike -= tt.sum(tt.log(tt.diag(cho_gp_cov)))
        lnlike -= 0.5 * N * tt.log(2 * np.pi)

        return lnlike[0, 0]

    @autocompile
    def lnlike_woodbury(self, X, flux, CInv, mu, LInv, lndetC, lndetL):
        """
        Compute the log marginal likelihood of the data given a design matrix
        using the Woodbury identity.

        Args:
            X (matrix): The flux design matrix.
            flux (array): The flux timeseries.
            CInv (scalar/vector/matrix): The inverse data covariance matrix.
            mu (array): The prior mean of the spherical harmonic coefficients.
            L (scalar/vector/matrix): The inverse prior covariance of the
                spherical harmonic coefficients.

        Returns:
            The log marginal likelihood of the `flux` vector conditioned on
            the design matrix `X`. This is the likelihood marginalized over
            all possible spherical harmonic vectors, which is analytically
            computable for the linear `starry` model.

        """
        # Compute the GP mean
        gp_mu = tt.dot(X, mu)

        # Residual vector
        r = tt.reshape(flux - gp_mu, (-1, 1))

        # Inverse of GP covariance via Woodbury identity
        if CInv.ndim == 0:
            U = X * CInv
        elif CInv.ndim == 1:
            U = tt.dot(tt.diag(CInv), X)
        else:
            U = tt.dot(CInv, X)

        if LInv.ndim == 0:
            W = tt.dot(tt.transpose(X), U) + LInv * tt.eye(U.shape[1])
        elif LInv.ndim == 1:
            W = tt.dot(tt.transpose(X), U) + tt.diag(LInv)
        else:
            W = tt.dot(tt.transpose(X), U) + LInv
        cho_W = sla.cholesky(W)

        if CInv.ndim == 0:
            SInv = CInv * tt.eye(U.shape[0]) - tt.dot(
                U, self.cho_solve(cho_W, tt.transpose(U))
            )
        elif CInv.ndim == 1:
            SInv = tt.diag(CInv) - tt.dot(
                U, self.cho_solve(cho_W, tt.transpose(U))
            )
        else:
            SInv = CInv - tt.dot(U, self.cho_solve(cho_W, tt.transpose(U)))

        # Determinant of GP covariance
        lndetW = 2 * tt.sum(tt.log(tt.diag(cho_W)))
        lndetS = lndetW + lndetC + lndetL

        # Compute the marginal likelihood
        N = X.shape[0]
        lnlike = -0.5 * tt.dot(tt.transpose(r), tt.dot(SInv, r))
        lnlike -= 0.5 * lndetS
        lnlike -= 0.5 * N * tt.log(2 * np.pi)

        return lnlike[0, 0]


# Instantiate the Op
linalg = OpsLinAlg()
linalg.Covariance = _Covariance
