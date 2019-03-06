# -*- coding: utf-8 -*-
try:
    from ._starry_extensions import *
except ImportError:
    pass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def MAP(A, L, C, flux, return_cov=False, return_cho_W=False):
    """

    """
    nt, nr = A.shape
    assert flux.shape == (nt,), "Invalid shape for `flux`."
    
    # Compute C^-1 . A
    if type(C) is tuple:
        CInvA = cho_solve(C, A)
    elif not hasattr(C, "__len__"):
        CInvA = (1.0 / C) * A
    elif C.shape == (1,):
        CInvA = (1.0 / C[0]) * A
    elif (C.shape == (nt,)):
        CInvA = (1.0 / C)[:, None] * A
    elif (C.shape == (nt, nt)):
        CInvA = np.linalg.solve(C, A)

    # Compute W = A^T . C^-1 . A + L^-1
    W = np.dot(A.T, CInvA)
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
    M = cho_solve(cho_W, CInvA.T)
    yhat = np.dot(M, flux)
    if return_cov:
        yvar = cho_solve(W, np.eye(nr))
        if return_cho_W:
            return yhat, yvar, cho_W
        else:
            return yhat, yvar
    else:
        if return_cho_W:
            return yhat, cho_W
        else:
            return yhat
    

def likelihood(A, L, C, flux, yhat=None):
    """

    """
    # Infer shapes
    nt, nr = A.shape

    # If `yhat` is not provided, compute the *marginal* likelihood
    if yhat is None:
        yhat, cho_W = MAP(A, L, C, flux, return_cho_W=True)
        logL = -np.sum(np.log(np.diag(cho_W[0])))
    else:
        logL = 0.0

    # Shape checks
    assert flux.shape == (nt,), "Invalid shape for `flux`."
    assert yhat.shape == (nr,), "Invalid shape for `yhat`."

    # Residual vector
    model = np.dot(A, yhat)
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
        wTLw = np.dot(r.T, cho_solve(L, r))
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


if __name__ == "__main__":
    # TODO: Make this into a unit test.
    nt = 10
    nr = 5
    A = np.random.randn(nt, nr)
    flux = np.random.randn(nt)
    C = 1.0
    L = 1.0
    yhat, yvar = MAP(A, L, C, flux)

    yhat_, yvar_ = MAP(A, np.ones(nr) * L, C, flux)
    assert np.allclose(yhat, yhat_)
    assert np.allclose(yvar, yvar_)

    yhat_, yvar_ = MAP(A, np.diag(np.ones(nr) * L), C, flux)
    assert np.allclose(yhat, yhat_)
    assert np.allclose(yvar, yvar_)

    yhat_, yvar_ = MAP(A, L, np.ones(nt) * C, flux)
    assert np.allclose(yhat, yhat_)
    assert np.allclose(yvar, yvar_)

    yhat_, yvar_ = MAP(A, L, np.diag(np.ones(nt)) * C, flux)
    assert np.allclose(yhat, yhat_)
    assert np.allclose(yvar, yvar_)