# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
import starry
import starry_beta
import pytest


def t3st_doppler():
    # Define all arguments
    kwargs = {
        "y":        [0.25, 0.25, 0.25],
        "theta":    30.0,
        "inc":      75.0,
        "obl":      30.0,
        "alpha":    0.40,
        "veq":      0.40,
        "xo":       0.15,
        "yo":       0.2,
        "zo":       1.0,
        "ro":       0.1,
        "u":        [0.4, 0.26]
    }
    theano_kwargs = {}
    for key in kwargs.keys():
        # NOTE: Use `theano.shared` instead of `as_tensor_variable` 
        # to prevent theano from treating variables whose values
        # are the same as the same variable! See
        # https://github.com/rodluger/starry/pull/195
        theano_kwargs[key] = theano.shared(np.float64(kwargs[key]), name=key)

    # Compute the rv and its gradient using starry
    map = starry.DopplerMap(ydeg=1, udeg=2)
    map.inc = kwargs.pop("inc")
    map.obl = kwargs.pop("obl")
    map.alpha = kwargs.pop("alpha")
    map.veq = kwargs.pop("veq")
    map[1:, :] = kwargs.pop("y")
    map[1:] = kwargs.pop("u")
    rv, grad = map.rv(gradient=True, **kwargs)

    # Instantiate the theano op
    model = map.rv_op(**theano_kwargs)

    # Compute the gradient using Theano
    varnames = sorted(theano_kwargs.keys())
    vars = [theano_kwargs[k] for k in varnames]
    computed = dict(zip(varnames,
                        theano.function([], theano.grad(model[0], vars))()))

    # Compare
    for key in theano_kwargs.keys():
        if key == "zo":
            continue
        assert np.allclose(np.squeeze(grad[key]), np.squeeze(computed[key]))


def test_limb_darkened():
    # Define all arguments
    npts = 3
    kwargs = {
        "u":        [0.4, 0.2],
        "b":        [0.15, 0.2, 0.25],
        "zo":       [1.0, 1.0, 1.0],
        "ro":       [0.1, 0.1, 0.1]
    }
    theano_kwargs = {}
    for key in kwargs.keys():
        theano_kwargs[key] = theano.shared(np.atleast_1d(kwargs[key]), name=key)
    udeg = len(kwargs["u"])

    # Compute the rv and its gradient using starry
    map_beta = starry_beta.Map(udeg)
    map_beta[1:] = kwargs.pop("u")
    kwargs["xo"] = kwargs.pop("b")
    kwargs.pop("zo")
    flux, grad = map_beta.flux(gradient=True, **kwargs)
    grad["b"] = grad.pop("xo")

    # Instantiate the theano op
    map = starry.Map(0, udeg)
    model = map.flux(**theano_kwargs)

    # Compute the gradient using Theano
    varnames = sorted(theano_kwargs.keys())
    vars = [theano_kwargs[k] for k in varnames]
    computed = {}
    for var in varnames:
        deriv = [theano.function([], 
                 theano.grad(model[i], theano_kwargs[var]))() 
                 for i in range(npts)]
        if var == "u":
            computed[var] = np.transpose(deriv)
        else:
            computed[var] = np.sum(deriv, axis=0)

    # Compare
    for key in theano_kwargs.keys():
        if key == "zo":
            continue
        assert np.allclose(np.squeeze(grad[key]), np.squeeze(computed[key]))


def test_limb_darkened_spectral():
    # Define all arguments
    npts = 3
    kwargs = {
        "u":        [[0.4, 0.2], [0.26, 0.13]],
        "b":        [0.15, 0.2, 0.25],
        "zo":       [1.0, 1.0, 1.0],
        "ro":       [0.1, 0.1, 0.1]
    }
    theano_kwargs = {}
    for key in kwargs.keys():
        theano_kwargs[key] = theano.shared(np.atleast_1d(kwargs[key]), name=key)
    udeg = len(kwargs["u"])

    # Compute the rv and its gradient using starry
    map_beta = starry_beta.Map(udeg, nwav=2)
    map_beta[1:] = kwargs.pop("u")
    kwargs["xo"] = kwargs.pop("b")
    kwargs.pop("zo")
    flux, grad = map_beta.flux(gradient=True, **kwargs)
    grad["b"] = grad.pop("xo")

    # Instantiate the theano op
    map = starry.Map(0, udeg, nw=2)
    model = map.flux(**theano_kwargs)

    # Compute the gradient using Theano
    varnames = sorted(theano_kwargs.keys())
    vars = [theano_kwargs[k] for k in varnames]
    computed = {}
    for var in varnames:
        if var == "u":
            deriv = np.array([[theano.function([], 
                               theano.grad(model[i, j], theano_kwargs[var]))() 
                               for j in range(udeg)] 
                               for i in range(npts)])
            computed[var] = np.sum(deriv, axis=1).swapaxes(0, 1)
        else:
            computed[var] = np.array([[theano.function([], 
                                       theano.grad(model[i, j], 
                                       theano_kwargs[var]))()[i] 
                                       for j in range(udeg)] 
                                       for i in range(npts)])

    # Compare
    for key in theano_kwargs.keys():
        if key == "zo":
            continue
        assert np.allclose(np.squeeze(grad[key]), np.squeeze(computed[key]))