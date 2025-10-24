# -*- coding: utf-8 -*-
from .. import config
from ..compat import pytensor, pt, ps, slinalg, floatX
from .._constants import *
from .utils import *
import numpy as np
from scipy.linalg import block_diag as scipy_block_diag
import scipy
from scipy.sparse import issparse, csr_matrix

__all__ = ["lazy_math", "greedy_math", "lazy_linalg", "greedy_linalg", "nadam"]


# Cholesky solve
try:
    # theano
    _solve_lower = slinalg.Solve(A_structure="lower_triangular", lower=True)
    _solve_upper = slinalg.Solve(A_structure="upper_triangular", lower=False)
except TypeError:
    # aesara/pytensor (TODO: Check me)
    _solve_lower = slinalg.Solve(assume_a="sym", lower=True)
    _solve_upper = slinalg.Solve(assume_a="sym", lower=False)


def _cho_solve(cho_A, b):
    return _solve_upper(pt.transpose(cho_A), _solve_lower(cho_A, b))


def _get_covariance(math, linalg, C=None, cho_C=None, N=None):
    """A container for covariance matrices.

    Args:
        C (scalar, vector, or matrix, optional): The covariance.
            Defaults to None.
        cho_C (matrix, optional): The lower Cholesky factorization of
            the covariance. Defaults to None.
        N (int, optional): The number of rows/columns in the covariance
            matrix, required if ``C`` is a scalar. Defaults to None.
    """

    # User provided the Cholesky factorization
    if cho_C is not None:

        cholesky = math.cast(cho_C)
        value = math.dot(cholesky, math.transpose(cholesky))
        inverse = linalg.cho_solve(cholesky, math.eye(cholesky.shape[0]))
        lndet = 2 * math.sum(math.log(math.diag(cholesky)))
        kind = "cholesky"
        N = cho_C.shape[0]

    # User provided the covariance as a scalar, vector, or matrix
    elif C is not None:

        C = math.cast(C)

        if hasattr(C, "ndim"):

            if C.ndim == 0:

                assert N is not None, "Please provide a matrix size `N`."
                cholesky = math.sqrt(C)
                inverse = math.cast(1.0 / C)
                lndet = math.cast(N * math.log(C))
                value = C
                kind = "scalar"

            elif C.ndim == 1:

                cholesky = math.sqrt(C)
                inverse = 1.0 / C
                lndet = math.sum(math.log(C))
                value = C
                kind = "vector"
                N = C.shape[0]

            else:

                cholesky = math.cholesky(C)
                inverse = linalg.cho_solve(cholesky, math.eye(C.shape[0]))
                lndet = 2 * math.sum(math.log(math.diag(cholesky)))
                value = C
                kind = "matrix"
                N = C.shape[0]

        # Assume it's a scalar
        else:

            assert N is not None, "Please provide a matrix size `N`."
            cholesky = math.sqrt(C)
            inverse = math.cast(1.0 / C)
            lndet = math.cast(N * math.log(C))
            value = C
            kind = "scalar"

    # ?!
    else:
        raise ValueError(
            "Either the covariance or its Cholesky factorization must be provided."
        )

    return value, cholesky, inverse, lndet, kind, N


class MathType(type):
    """Wrapper for pytensor/numpy functions."""

    def cholesky(cls, *args, **kwargs):
        if cls.lazy:
            return slinalg.cholesky(*args, **kwargs)
        else:
            return scipy.linalg.cholesky(*args, **kwargs, lower=True)

    def atleast_2d(cls, arg):
        if cls.lazy:
            return arg * pt.ones((1, 1))
        else:
            return np.atleast_2d(arg)

    def vectorize(cls, *args):
        """
        Vectorize all ``args`` so that they have the same length
        along the first axis.

        TODO: Add error catching if the dimensions don't agree.
        """
        if cls.lazy:
            args = [arg * pt.ones(1) for arg in args]
            size = pt.max([arg.shape[0] for arg in args])
            args = [pt.repeat(arg, size // arg.shape[0], 0) for arg in args]
        else:
            args = [np.atleast_1d(arg) for arg in args]
            size = np.max([arg.shape[0] for arg in args])
            args = tuple(
                [
                    arg
                    * np.ones(
                        (size,) + tuple(np.ones(len(arg.shape) - 1, dtype=int))
                    )
                    for arg in args
                ]
            )
        if len(args) == 1:
            return args[0]
        else:
            return args

    def cross(cls, x, y):
        """Cross product of two 3-vectors.

        Based on ``https://github.com/Theano/Theano/pull/3008``
        """
        if cls.lazy:
            eijk = np.zeros((3, 3, 3))
            eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
            eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
            return pt.as_tensor_variable(pt.dot(pt.dot(eijk, y), x))
        else:
            return np.cross(x, y)

    def cast(cls, *args):
        if cls.lazy:
            return cls.to_tensor(*args)
        else:
            if len(args) == 1:
                return np.array(args[0], dtype=floatX)
            else:
                return [np.array(arg, dtype=floatX) for arg in args]

    def sparse_cast(cls, *args):
        if cls.lazy:
            return cls.to_sparse_tensor(*args)
        else:
            if len(args) == 1:
                return csr_matrix(args[0], dtype=floatX)
            else:
                return [csr_matrix(arg, dtype=floatX) for arg in args]

    def to_array_or_tensor(cls, x):
        if cls.lazy:
            return pt.as_tensor_variable(x)
        else:
            return np.array(x)

    def block_diag(cls, *mats):
        if cls.lazy:
            N = [mat.shape[0] for mat in mats]
            Nsum = pt.sum(N)
            res = pt.zeros((Nsum, Nsum), dtype=floatX)
            n = 0
            for mat in mats:
                inds = slice(n, n + mat.shape[0])
                res = pt.set_subtensor(res[tuple((inds, inds))], mat)
                n += mat.shape[0]
            return res
        else:
            return scipy_block_diag(*mats)

    def to_tensor(cls, *args):
        """Convert all ``args`` to PyTensor tensor variables.

        Converts to tensor regardless of whether `cls.lazy` is True or False.
        """
        if len(args) == 1:
            return pt.as_tensor_variable(args[0]).astype(floatX)
        else:
            return [pt.as_tensor_variable(arg).astype(floatX) for arg in args]

    def to_sparse_tensor(cls, *args):
        """Convert all ``args`` to PyTensor sparse tensor variables.

        Converts to tensor regardless of whether `cls.lazy` is True or False.
        """
        if len(args) == 1:
            return ps.as_sparse_variable(args[0]).astype(floatX)
        else:
            return [ps.as_sparse_variable(arg).astype(floatX) for arg in args]

    def sparse_dot(cls, A, B):
        if cls.lazy:
            return ps.dot(A, B)
        else:
            if issparse(A):
                return A.dot(B)
            elif issparse(B):
                return (B.T.dot(A.T)).T
            else:
                raise ValueError("At least one input must be sparse.")

    def __getattr__(cls, attr):
        if cls.lazy:
            return getattr(tt, attr)
        else:
            return getattr(np, attr)


class LinAlgType(type):
    """Linear algebra operations."""

    @autocompile
    def cho_solve(self, cho_A, b):
        return _cho_solve(cho_A, b)

    @autocompile
    def solve(self, X, flux, cho_C, mu, LInv):
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
            CInvX = pt.dot(pt.diag(1 / cho_C ** 2), X)
        else:
            CInvX = _cho_solve(cho_C, X)

        # Compute W = X^T . C^-1 . X + L^-1
        W = pt.dot(pt.transpose(X), CInvX)
        if LInv.ndim == 0:
            W = pt.inc_subtensor(
                W[tuple((pt.arange(W.shape[0]), pt.arange(W.shape[0])))], LInv
            )
            LInvmu = mu * LInv
        elif LInv.ndim == 1:
            W = pt.inc_subtensor(
                W[tuple((pt.arange(W.shape[0]), pt.arange(W.shape[0])))], LInv
            )
            LInvmu = mu * LInv
        else:
            W += LInv
            LInvmu = pt.dot(LInv, mu)

        # Compute the max like y and its covariance matrix
        cho_W = slinalg.cholesky(W)
        M = _cho_solve(cho_W, pt.transpose(CInvX))
        yhat = pt.dot(M, flux) + _cho_solve(cho_W, LInvmu)
        ycov = _cho_solve(cho_W, pt.eye(cho_W.shape[0]))
        cho_ycov = slinalg.cholesky(ycov)

        return yhat, cho_ycov

    @autocompile
    def lnlike(cls, X, flux, C, mu, L):
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
        gp_mu = pt.dot(X, mu)

        # Compute the GP covariance
        if L.ndim == 0:
            XLX = pt.dot(X, pt.transpose(X)) * L
        elif L.ndim == 1:
            XLX = pt.dot(pt.dot(X, pt.diag(L)), pt.transpose(X))
        else:
            XLX = pt.dot(pt.dot(X, L), pt.transpose(X))

        if C.ndim == 0 or C.ndim == 1:
            gp_cov = pt.inc_subtensor(
                XLX[tuple((pt.arange(XLX.shape[0]), pt.arange(XLX.shape[0])))],
                C,
            )
        else:
            gp_cov = C + XLX

        cho_gp_cov = slinalg.cholesky(gp_cov)

        # Compute the marginal likelihood
        N = X.shape[0]
        r = pt.reshape(flux - gp_mu, (-1, 1))
        lnlike = -0.5 * pt.dot(pt.transpose(r), _cho_solve(cho_gp_cov, r))
        lnlike -= pt.sum(pt.log(pt.diag(cho_gp_cov)))
        lnlike -= 0.5 * N * pt.log(2 * np.pi)

        return lnlike[0, 0]

    @autocompile
    def lnlike_woodbury(cls, X, flux, CInv, mu, LInv, lndetC, lndetL):
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
        gp_mu = pt.dot(X, mu)

        # Residual vector
        r = pt.reshape(flux - gp_mu, (-1, 1))

        # Inverse of GP covariance via Woodbury identity
        if CInv.ndim == 0:
            U = X * CInv
        elif CInv.ndim == 1:
            U = pt.dot(pt.diag(CInv), X)
        else:
            U = pt.dot(CInv, X)

        if LInv.ndim == 0:
            W = pt.dot(pt.transpose(X), U) + LInv * pt.eye(U.shape[1])
        elif LInv.ndim == 1:
            W = pt.dot(pt.transpose(X), U) + pt.diag(LInv)
        else:
            W = pt.dot(pt.transpose(X), U) + LInv
        cho_W = slinalg.cholesky(W)

        if CInv.ndim == 0:
            SInv = CInv * pt.eye(U.shape[0]) - pt.dot(
                U, _cho_solve(cho_W, pt.transpose(U))
            )
        elif CInv.ndim == 1:
            SInv = pt.diag(CInv) - pt.dot(
                U, _cho_solve(cho_W, pt.transpose(U))
            )
        else:
            SInv = CInv - pt.dot(U, _cho_solve(cho_W, pt.transpose(U)))

        # Determinant of GP covariance
        lndetW = 2 * pt.sum(pt.log(pt.diag(cho_W)))
        lndetS = lndetW + lndetC + lndetL

        # Compute the marginal likelihood
        N = X.shape[0]
        lnlike = -0.5 * pt.dot(pt.transpose(r), pt.dot(SInv, r))
        lnlike -= 0.5 * lndetS
        lnlike -= 0.5 * N * pt.log(2 * np.pi)

        return lnlike[0, 0]

    @autocompile
    def _cho_solve(cls, cho_A, b):
        return _cho_solve(cho_A, b)


class lazy_math(metaclass=MathType):
    """Alias for ``numpy`` or ``pytensor.tensor``."""

    lazy = True


class greedy_math(metaclass=MathType):
    """Alias for ``numpy`` or ``pytensor.tensor``."""

    lazy = False


class lazy_linalg(metaclass=LinAlgType):
    """Miscellaneous linear algebra operations."""

    lazy = True

    class Covariance(object):
        def __init__(self, *args, **kwargs):
            (
                self.value,
                self.cholesky,
                self.inverse,
                self.lndet,
                self.kind,
                self.N,
            ) = _get_covariance(lazy_math, lazy_linalg, *args, **kwargs)


class greedy_linalg(metaclass=LinAlgType):
    """Miscellaneous linear algebra operations."""

    lazy = False

    class Covariance(object):
        def __init__(self, *args, **kwargs):
            (
                self.value,
                self.cholesky,
                self.inverse,
                self.lndet,
                self.kind,
                self.N,
            ) = _get_covariance(greedy_math, greedy_linalg, *args, **kwargs)


def nadam(cost, params, lr=0.002, b1=0.9, b2=0.999, e=1e-8, sd=0.004):
    """
    Optimizer that implements the NAdam algorithm.

    Args:
        lr: The learning rate.
        b1: The exponential decay rate for the 1st moment estimates.
        b2: The exponential decay rate for the exponentially
            weighted infinity norm.
        e: A small constant for numerical stability.
        sd: Schedule decay.

    Adapted from
    https://github.com/keras-team/keras/blob/master/keras/optimizers.py

    """
    updates = []
    grads = pt.grad(cost, params)
    i = pytensor.shared(np.array(0.0, dtype=pytensor.config.floatX))
    i_t = i + 1.0

    # Warm up
    m_schedule = pytensor.shared(np.array(1.0, dtype=pytensor.config.floatX))
    momentum_cache_t = b1 * (1.0 - 0.5 * (pt.pow(0.96, i_t * sd)))
    momentum_cache_t_1 = b1 * (1.0 - 0.5 * (pt.pow(0.96, (i_t + 1) * sd)))
    m_schedule_new = m_schedule * momentum_cache_t
    m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
    updates.append((m_schedule, m_schedule_new))

    for p, g in zip(params, grads):
        m = pytensor.shared(p.get_value() * 0.0)
        v = pytensor.shared(p.get_value() * 0.0)

        g_prime = g / (1.0 - m_schedule_new)
        m_t = b1 * m + (1.0 - b1) * g
        m_t_prime = m_t / (1.0 - m_schedule_next)
        v_t = b2 * v + (1.0 - b2) * pt.sqr(g)
        v_t_prime = v_t / (1.0 - pt.pow(b2, i_t))
        m_t_bar = (1.0 - momentum_cache_t) * g_prime + (
            momentum_cache_t_1 * m_t_prime
        )

        updates.append((m, m_t))
        updates.append((v, v_t))

        p_t = p - lr * m_t_bar / (pt.sqrt(v_t_prime) + e)
        new_p = p_t
        updates.append((p, new_p))

    updates.append((i, i_t))

    return updates
