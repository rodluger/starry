# -*- coding: utf-8 -*-
from inspect import getmro
import theano
import theano.tensor as tt
import numpy as np


__all__ = ["is_theano",
           "to_tensor",
           "vectorize",
           "autoeval"]


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


def autoeval(func):
    """
    Magic wrapper to auto-evaluate functions if none
    of the arguments is a Theano variable.
    
    """
    def wrapper(*args, **kwargs):
        if is_theano(*args) or is_theano(*kwargs.values()):
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs).eval()
    return wrapper