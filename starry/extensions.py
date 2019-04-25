# -*- coding: utf-8 -*-
"""
This module provides custom extensions to :py:obj:`starry`.

.. autofunction:: MAP
.. autofunction:: log_likelihood
.. autofunction:: RAxisAngle(axis, angle)

"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from . import modules
try:
    from ._starry_extensions import RAxisAngle
except ImportError:
    def RAxisAngle(*args, **kwargs):
        bit = modules["_STXRRY_EXTENSIONS_"]
        raise ModuleNotFoundError("Please re-compile `starry` " + 
                                  "with bit %d enabled." % bit)


def MAP(X, L, C, flux, return_cov=False, return_cho_W=False):
    """
    Compute the maximum a posteriori (MAP) prediction for the
    spherical harmonic coefficients of a map given a flux timeseries.

    Args:
        X (ndarray): The design matrix returned by \
            :py:meth:`Map.linear_flux_model`, shape (ntime, nregressors).
        L: The prior covariance of the spherical harmonic coefficients. \
            This may be a scalar, a vector, a matrix, or the Cholesky \
            factorization of the covariance matrix (a tuple returned by \
            :py:obj:`scipy.linalg.cho_factor`).
        C: The data covariance. This may be a scalar, a vector, a matrix, \
            or the Cholesky factorization of the covariance matrix (a tuple \
            returned by :py:obj:`scipy.linalg.cho_factor`).
        flux (ndarray): The flux timeseries.
        return_cov (bool): Return the covariance matrix of the solution? Default :py:obj:`False`.
        return_cho_W (bool): Return the Cholesky factorization of the \
            quantity :math:`W = X^T C^{-1} X + L^{-1}`? Default :py:obj:`False`.

    Returns:
        The vector of spherical harmonic coefficients corresponding to the
        MAP solution, and optionally the covariance of the solution and the
        Cholesky factorization of :math:`W` (see above).

    """
    nt, nr = X.shape
    assert flux.shape == (nt,), "Invalid shape for `flux`."
    
    # Compute C^-1 . X
    if type(C) is tuple:
        CInvX = cho_solve(C, X)
    elif not hasattr(C, "__len__"):
        CInvX = (1.0 / C) * X
    elif C.shape == (1,):
        CInvX = (1.0 / C[0]) * X
    elif (C.shape == (nt,)):
        CInvX = (1.0 / C)[:, None] * X
    elif (C.shape == (nt, nt)):
        CInvX = np.linalg.solve(C, X)

    # Compute W = X^T . C^-1 . X + L^-1
    W = np.dot(X.T, CInvX)
    if type(L) is tuple:
        W += cho_solve(L, np.eye(nr))
    elif not hasattr(L, "__len__"):
        W[np.diag_indices_from(W)] += 1.0 / L
    elif L.shape == (1,):
        W[np.diag_indices_from(W)] += 1.0 / L[0]
    elif (L.shape == (nr,)):
        W[np.diag_indices_from(W)] += 1.0 / L
    elif (L.shape == (nr, nr)):
        W += np.linalg.inv(L)
    else:
        raise ValueError("Invalid shape for `L`.")

    # Compute the max like y and its covariance matrix
    cho_W = cho_factor(W)
    M = cho_solve(cho_W, CInvX.T)
    yhat = np.dot(M, flux)
    if return_cov:
        yvar = cho_solve(cho_W, np.eye(nr))
        if return_cho_W:
            return yhat, yvar, cho_W
        else:
            return yhat, yvar
    else:
        if return_cho_W:
            return yhat, cho_W
        else:
            return yhat
    

def log_likelihood(X, L, C, flux, yhat=None):
    """
    Compute the log likelihood of the linear model given
    a flux timeseries.

    Args:
        X (ndarray): The design matrix returned by \
            :py:meth:`Map.linear_flux_model`, shape (ntime, nregressors).
        L: The prior covariance of the spherical harmonic coefficients. \
            This may be a scalar, a vector, a matrix, or the Cholesky \
            factorization of the covariance matrix (a tuple returned by \
            :py:obj:`scipy.linalg.cho_factor`).
        C: The data covariance. This may be a scalar, a vector, a matrix, \
            or the Cholesky factorization of the covariance matrix (a tuple \
            returned by :py:obj:`scipy.linalg.cho_factor`).
        flux (ndarray): The flux timeseries.
        yhat (ndarray): If not :py:obj:`None`, computes the log likelihood \
            of the model specified by the spherical harmonic vector :py:obj:`yhat`. If \
            :py:obj:`yhat` is not provided, instead computes the *marginal \
            log likelihood* of the model, marginalizing over all values \
            of :py:obj:`yhat`.

    Returns: 
        The log likelihood (optionally the marginal log likelihood), a scalar.

    """
    # Infer shapes
    nt, nr = X.shape

    # If `yhat` is not provided, compute the *marginal* likelihood
    if yhat is None:
        yhat, cho_W = MAP(X, L, C, flux, return_cho_W=True)
        logL = -np.sum(np.log(np.diag(cho_W[0])))
    else:
        logL = 0.0

    # Shape checks
    assert flux.shape == (nt,), "Invalid shape for `flux`."
    assert yhat.shape == (nr,), "Invalid shape for `yhat`."

    # Residual vector
    model = np.dot(X, yhat)
    r = flux - model

    # Data likelihood
    if type(C) is tuple:
        rTCr = np.dot(r.T, cho_solve(C, r))
    elif not hasattr(C, "__len__"):
         rTCr = (1.0 / C) * np.dot(r.T, r)
    elif (C.shape == (1,)):
        rTCr = (1.0 / C[0]) * np.dot(r.T, r)
    elif (C.shape == (nt,)):
        rTCr = np.dot(r.T, r / C)
    elif (C.shape == (nt, nt)):
        rTCr = np.dot(r.T, np.linalg.solve(C, r))
    else:
        raise ValueError("Invalid shape for `C`.")
    logL -= 0.5 * rTCr

    # L2 penalty
    if type(L) is tuple:
        wTLw = np.dot(yhat.T, cho_solve(L, yhat))
    elif not hasattr(L, "__len__"):
        wTLw = (1.0 / L) * np.dot(yhat.T, yhat)
    elif L.shape == (1,):
        wTLw = (1.0 / L[0]) * np.dot(yhat.T, yhat)
    elif (L.shape == (nr,)):
        wTLw = np.dot(yhat.T, yhat / L)
    elif (L.shape == (nr, nr)):
        wTLw = np.dot(yhat.T, np.linalg.solve(L, yhat))
    else:
        raise ValueError("Invalid shape for `L`.")
    logL -= 0.5 * wTLw

    # Determinant terms
    for S in [L, C]:
        if type(S) is tuple:
            logL -= np.sum(np.log(S[0]))
        elif not hasattr(S, "__len__"):
            logL -= 0.5 * nr * np.log(S)
        elif (S.shape == (1,)):
            logL -= 0.5 * nr * np.log(S[0])
        elif (S.shape == (nr,)):
            logL -= 0.5 * np.sum(np.log(S))
        elif (S.shape == (nr, nr)):
            logL -= 0.5 * np.log(np.linalg.det(S))
        else:
            raise ValueError("One of the matrices has the wrong shape.")
    
    return logL