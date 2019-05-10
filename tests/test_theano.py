# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
import starry
import starry_beta
import pytest


def run(map=None, func=None, **kwargs):
    """

    """
    # Create Theano tensors
    theano_kwargs = {}
    for key in kwargs.keys():
        theano_kwargs[key] = theano.shared(np.float64(kwargs[key]), name=key)

    # Compute the flattened model
    model = tt.flatten(getattr(map, func)(**theano_kwargs))

    # Compute the gradient using Theano
    varnames = sorted(theano_kwargs.keys())
    vars = [theano_kwargs[k] for k in varnames]

    # Loop over all indices of the model
    for k in range(len(model.shape.eval())):
        grad = dict(zip(varnames,
                        theano.function([], 
                        theano.grad(model[k], vars))()))

        # Fudge some shapes
        if grad["u"].ndim == 2:
            grad["u"] = grad["u"][:, k]

        # Compute the gradient numerically
        eps = 1e-8
        grad_num = {}

        for key in varnames:
            if key in ["y", "u"]:
                N = map.udeg if key == "u" else map.Ny - 1
                grad_num[key] = np.zeros(N)
                for i in range(N):
                    if map._limbdarkened and map._spectral:
                        inds = tuple((i, k))
                    else:
                        inds = i
                    val = kwargs[key][inds]
                    kwargs[key][inds] = val - eps
                    f1 = np.atleast_1d(getattr(map, func)(**kwargs))[k]
                    kwargs[key][inds] = val + eps
                    f2 = np.atleast_1d(getattr(map, func)(**kwargs))[k]
                    kwargs[key][inds] = val
                    grad_num[key][i] = (f2 - f1) / (2 * eps)
            else:
                val = kwargs[key]
                kwargs[key] = val - eps
                f1 = np.atleast_1d(getattr(map, func)(**kwargs))[k]
                kwargs[key] = val + eps
                f2 = np.atleast_1d(getattr(map, func)(**kwargs))[k]
                kwargs[key] = val
                grad_num[key] = (f2 - f1) / (2 * eps)

        # Compare
        for key in varnames:
            try:
                assert np.allclose(grad[key], grad_num[key], atol=1e-5, rtol=1e-5)
            except:
                print("Mismatch in %s:" % (key))
                print("Expected ", grad_num[key])
                print("Got      ", grad[key])
                assert np.allclose(grad[key], grad_num[key], atol=1e-5, rtol=1e-5)


def test_doppler():
    kwargs = {
        "map":      starry.Map(ydeg=1, udeg=2, doppler=True),
        "func":     "rv",
        "y":        np.array([0.25, 0.25, 0.25]),
        "u":        np.array([0.4, 0.26]),
        "theta":    30.0,
        "inc":      75.0,
        "obl":      30.0,
        "alpha":    0.40,
        "veq":      0.40,
        "xo":       0.15,
        "yo":       0.2,
        "zo":       1.0,
        "ro":       0.1
    }
    run(**kwargs)


def test_ylm():
    kwargs = {
        "map":      starry.Map(ydeg=1, udeg=2),
        "func":     "flux",
        "y":        np.array([0.25, 0.25, 0.25]),
        "u":        np.array([0.4, 0.26]),
        "theta":    30.0,
        "inc":      75.0,
        "obl":      30.0,
        "xo":       0.15,
        "yo":       0.2,
        "zo":       1.0,
        "ro":       0.1
    }
    run(**kwargs)


def test_ld():
    kwargs = {
        "map":      starry.Map(udeg=2),
        "func":     "flux",
        "u":        np.array([0.4, 0.26]),
        "b":        0.15,
        "zo":       1.0,
        "ro":       0.1
    }
    run(**kwargs)


def test_ld_spectral():
    """
    TODO: The limb darkening derivs are always off
    by exactly a factor of two. Not sure why.
    """
    kwargs = {
        "map":      starry.Map(udeg=3, nw=3),
        "func":     "flux",
        "u":        np.array([[0.8, 0.2, 0.4], 
                              [0.1, 0.13, 0.4], 
                              [0.3, 0.3, 0.3]]),
        "b":        0.15,
        "zo":       1.0,
        "ro":       0.1
    }
    run(**kwargs)


if __name__ == "__main__":
    test_doppler()
    test_ylm()
    test_ld()
    test_ld_spectral()