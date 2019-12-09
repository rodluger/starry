# -*- coding: utf-8 -*-
from .. import config
from ..constants import *
import theano
import theano.tensor as tt
import numpy as np
from scipy.linalg import block_diag as scipy_block_diag
import theano.tensor.slinalg as sla
import scipy

__all__ = ["math"]


class MathType(type):
    """Wrapper for theano/numpy functions."""

    def cholesky(cls, *args, **kwargs):
        if config.lazy:
            return sla.cholesky(*args, **kwargs)
        else:
            return scipy.linalg.cholesky(*args, **kwargs, lower=True)

    def atleast_2d(cls, arg):
        if config.lazy:
            return arg * tt.ones((1, 1))
        else:
            return np.atleast_2d(arg)

    def vectorize(cls, *args):
        """
        Vectorize all ``args`` so that they have the same length
        along the first axis.

        TODO: Add error catching if the dimensions don't agree.
        """
        if config.lazy:
            args = [arg * tt.ones(1) for arg in args]
            size = tt.max([arg.shape[0] for arg in args])
            args = [tt.repeat(arg, size // arg.shape[0], 0) for arg in args]
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

    def cross(x, y):
        """Cross product of two 3-vectors.

        Based on ``https://github.com/Theano/Theano/pull/3008``
        """
        if config.lazy:
            eijk = np.zeros((3, 3, 3))
            eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
            eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
            return tt.as_tensor_variable(tt.dot(tt.dot(eijk, y), x))
        else:
            return np.cross(x, y)

    def cast(cls, *args):
        if config.lazy:
            return cls.to_tensor(*args)
        else:
            if len(args) == 1:
                return np.array(args[0], dtype=tt.config.floatX)
            else:
                return [np.array(arg, dtype=tt.config.floatX) for arg in args]

    def to_array_or_tensor(cls, x):
        if config.lazy:
            return tt.as_tensor_variable(x)
        else:
            return np.array(x)

    def block_diag(self, *mats):
        if config.lazy:
            N = [mat.shape[0] for mat in mats]
            Nsum = tt.sum(N)
            res = tt.zeros((Nsum, Nsum), dtype=theano.config.floatX)
            n = 0
            for mat in mats:
                inds = slice(n, n + mat.shape[0])
                res = tt.set_subtensor(res[tuple((inds, inds))], mat)
                n += mat.shape[0]
            return res
        else:
            return scipy_block_diag(*mats)

    def to_tensor(cls, *args):
        """Convert all ``args`` to Theano tensor variables.

        Converts to tensor regardless of whether `config.lazy` is True or False.
        """
        if len(args) == 1:
            return tt.as_tensor_variable(args[0]).astype(tt.config.floatX)
        else:
            return [
                tt.as_tensor_variable(arg).astype(tt.config.floatX)
                for arg in args
            ]

    def __getattr__(cls, attr):
        if config.lazy:
            return getattr(tt, attr)
        else:
            return getattr(np, attr)


class math(metaclass=MathType):
    """Alias for ``numpy`` or ``theano.tensor``, depending on `config.lazy`."""

    pass
