# -*- coding: utf-8 -*-
from ._core import math
from . import config
import numpy as np


def solve(
    design_matrix,
    data,
    *,
    C=None,
    cho_C=None,
    mu=0.0,
    L=None,
    cho_L=None,
    N=None,
    lazy=None,
):
    """
    Solve the generalized least squares (GLS) problem.

    Args:
        design_matrix (matrix): The design matrix that transforms a vector
            from coefficient space to data space.
        data (vector): The observed dataset.
        C (scalar, vector, or matrix): The data covariance. This may be
            a scalar, in which case the noise is assumed to be
            homoscedastic, a vector, in which case the covariance
            is assumed to be diagonal, or a matrix specifying the full
            covariance of the dataset. Default is None. Either `C` or
            `cho_C` must be provided.
        cho_C (matrix): The lower Cholesky factorization of the data
            covariance matrix. Defaults to None. Either `C` or
            `cho_C` must be provided.
        mu (scalar or vector): The prior mean on the regression coefficients.
            Default is zero.
        L (scalar, vector, or matrix): The prior covariance. This may be
            a scalar, in which case the covariance is assumed to be
            homoscedastic, a vector, in which case the covariance
            is assumed to be diagonal, or a matrix specifying the full
            prior covariance. Default is None. Either `L` or
            `cho_L` must be provided.
        cho_L (matrix): The lower Cholesky factorization of the prior
            covariance matrix. Defaults to None. Either `L` or
            `cho_L` must be provided.
        N (int, optional): The number of regression coefficients. This is
            necessary only if both ``mu`` and ``L`` are provided as scalars.

    Returns:
        A tuple containing the posterior mean for the regression \
        coefficients (a vector) and the Cholesky factorization \
        of the posterior covariance (a lower triangular matrix).

    """
    if lazy is None:
        lazy = config.lazy
    if lazy:
        _math = math.lazy_math
        _linalg = math.lazy_linalg
    else:
        _math = math.greedy_math
        _linalg = math.greedy_linalg

    design_matrix = _math.cast(design_matrix)
    data = _math.cast(data)
    C = _linalg.Covariance(C, cho_C, N=data.shape[0])
    mu = _math.cast(mu)
    if L is None and cho_L is None:
        raise ValueError(
            "Either the prior covariance or its "
            "Cholesky factorization must be provided."
        )
    elif L is not None:
        tmp = _math.cast(L)
    else:
        tmp = _math.cast(cho_L)
    if mu.ndim == 0 and tmp.ndim == 0:
        assert (
            N is not None
        ), "Please provide the number of coefficients ``N``."
    elif mu.ndim > 0:
        N = mu.shape[0]
    elif L.ndim > 0:
        N = tmp.shape[0]
    if mu.ndim == 0:
        mu = mu * _math.ones(N)
    L = _linalg.Covariance(L, cho_L, N=N)
    return _linalg.solve(design_matrix, data, C.cholesky, mu, L.inverse)


def lnlike(
    design_matrix,
    data,
    *,
    C=None,
    cho_C=None,
    mu=0.0,
    L=None,
    cho_L=None,
    N=None,
    woodbury=True,
    lazy=None,
):
    """
    Compute the log marginal likelihood of the data given a design matrix.

    Args:
        design_matrix (matrix): The design matrix that transforms a vector
            from coefficient space to data space.
        data (vector): The observed dataset.
        C (scalar, vector, or matrix): The data covariance. This may be
            a scalar, in which case the noise is assumed to be
            homoscedastic, a vector, in which case the covariance
            is assumed to be diagonal, or a matrix specifying the full
            covariance of the dataset. Default is None. Either `C` or
            `cho_C` must be provided.
        cho_C (matrix): The lower Cholesky factorization of the data
            covariance matrix. Defaults to None. Either `C` or
            `cho_C` must be provided.
        mu (scalar or vector): The prior mean on the regression coefficients.
            Default is zero.
        L (scalar, vector, or matrix): The prior covariance. This may be
            a scalar, in which case the covariance is assumed to be
            homoscedastic, a vector, in which case the covariance
            is assumed to be diagonal, or a matrix specifying the full
            prior covariance. Default is None. Either `L` or
            `cho_L` must be provided.
        cho_L (matrix): The lower Cholesky factorization of the prior
            covariance matrix. Defaults to None. Either `L` or
            `cho_L` must be provided.
        N (int, optional): The number of regression coefficients. This is
            necessary only if both ``mu`` and ``L`` are provided as scalars.
        woodbury (bool, optional): Solve the linear problem using the
            Woodbury identity? Default is True. The
            `Woodbury identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_
            is used to speed up matrix operations in the case that the
            number of data points is much larger than the number of
            regression coefficients. In this limit, it can
            speed up the code by more than an order of magnitude. Keep
            in mind that the numerical stability of the Woodbury identity
            is not great, so if you're getting strange results try
            disabling this. It's also a good idea to disable this in the
            limit of few data points and large number of regressors.

    Returns:
        The log marginal likelihood, a scalar.

    """
    if lazy is None:
        lazy = config.lazy
    if lazy:
        _math = math.lazy_math
        _linalg = math.lazy_linalg
    else:
        _math = math.greedy_math
        _linalg = math.greedy_linalg

    design_matrix = _math.cast(design_matrix)
    data = _math.cast(data)
    C = _linalg.Covariance(C, cho_C, N=data.shape[0])
    mu = _math.cast(mu)
    if L is None and cho_L is None:
        raise ValueError(
            "Either the prior covariance or its "
            "Cholesky factorization must be provided."
        )
    elif L is not None:
        tmp = _math.cast(L)
    else:
        tmp = _math.cast(cho_L)
    if mu.ndim == 0 and tmp.ndim == 0:
        assert (
            N is not None
        ), "Please provide the number of coefficients ``N``."
    elif mu.ndim > 0:
        N = mu.shape[0]
    elif L.ndim > 0:
        N = tmp.shape[0]
    if mu.ndim == 0:
        mu = mu * _math.ones(N)
    L = _linalg.Covariance(L, cho_L, N=N)
    if woodbury:
        return _linalg.lnlike_woodbury(
            design_matrix, data, C.inverse, mu, L.inverse, C.lndet, L.lndet
        )
    else:
        return _linalg.lnlike(design_matrix, data, C.value, mu, L.value)
