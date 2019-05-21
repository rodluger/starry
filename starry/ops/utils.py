# -*- coding: utf-8 -*-
from inspect import getmro
import theano
import theano.tensor as tt
import numpy as np


__all__ = ["is_theano",
           "to_tensor",
           "vectorize",
           "cross",
           "RAxisAngle",
           "VectorRAxisAngle"]


def is_theano(*objs):
    """
    Return ``True`` if any of ``objs`` is a ``Theano`` object.

    """
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


def to_tensor(*args):
    """
    Convert all ``args`` to tensor variables.

    """
    if len(args) == 1:
        return tt.as_tensor_variable(args[0]).astype(tt.config.floatX)
    else:
        return [tt.as_tensor_variable(arg).astype(tt.config.floatX) for arg in args]


def vectorize(*args):
    """
    Vectorize all scalar ``args``.

    """
    if is_theano(*args):
        ones = tt.ones_like(np.sum([arg if is_theano(arg) 
                                    else np.atleast_1d(arg) 
                                    for arg in args], axis=0)).astype(tt.config.floatX).reshape([-1])
        args = tuple([to_tensor(arg).astype(tt.config.floatX) * ones for arg in args])
    else:
        ones = np.ones_like(np.sum([np.atleast_1d(arg) for arg in args], axis=0))
        args = tuple([arg * ones for arg in args])
    return args


def cross(x, y):
    """
    Cross product of two 3-vectors.

    Based on ``https://github.com/Theano/Theano/pull/3008``
    """
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    result = tt.as_tensor_variable(
        tt.dot(tt.dot(eijk, y), x)
    )
    return result


def RAxisAngle(axis=[0, 1, 0], theta=0):
    """

    """
    cost = tt.cos(theta * np.pi / 180.)
    sint = tt.sin(theta * np.pi / 180.)

    return tt.reshape(tt.as_tensor_variable([
        cost + axis[0] * axis[0] * (1 - cost),
        axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
        axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
        axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
        cost + axis[1] * axis[1] * (1 - cost),
        axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
        axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
        axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
        cost + axis[2] * axis[2] * (1 - cost)
    ]), [3, 3])


def VectorRAxisAngle(axis=[0, 1, 0], theta=0):
    """

    """
    fn = lambda theta, axis: RAxisAngle(axis=axis, theta=theta)
    R, _ = theano.scan(fn=fn, sequences=[theta], non_sequences=[axis])
    return R