# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
import starry
import starry_beta
import pytest


def test_doppler():
    """

    """
    kwargs = {
        "y":        np.array([0.25, 0.25, 0.25]),
        "theta":    30.0,
        "inc":      75.0,
        "obl":      30.0,
        "alpha":    0.40,
        "veq":      0.40,
        "xo":       0.15,
        "yo":       0.2,
        "zo":       1.0,
        "ro":       0.1,
        "u":        np.array([0.4, 0.26])
    }
    theano_kwargs = {}
    for key in kwargs.keys():
        theano_kwargs[key] = theano.shared(np.float64(kwargs[key]), name=key)

    # Instantiate the map
    map = starry.Map(ydeg=1, udeg=2, doppler=True)
    model = map.rv(**theano_kwargs)

    # Compute the gradient using Theano
    varnames = sorted(theano_kwargs.keys())
    vars = [theano_kwargs[k] for k in varnames]
    grad = dict(zip(varnames,
                    theano.function([], 
                    theano.grad(model[0], vars))()))

    # Compute the gradient numerically
    eps = 1e-8
    grad_num = {}
    for key in varnames:
        if key in ["y", "u"]:
            grad_num[key] = np.zeros_like(kwargs[key])
            for i in range(len(kwargs[key])):
                val = kwargs[key][i]
                kwargs[key][i] = val - eps
                f1 = map.rv(**kwargs)
                kwargs[key][i] = val + eps
                f2 = map.rv(**kwargs)
                kwargs[key][i] = val
                grad_num[key][i] = (f2 - f1) / (2 * eps)
        else:
            val = kwargs[key]
            kwargs[key] = val - eps
            f1 = map.rv(**kwargs)
            kwargs[key] = val + eps
            f2 = map.rv(**kwargs)
            kwargs[key] = val
            grad_num[key] = np.squeeze((f2 - f1) / (2 * eps))

    # Compare
    for key in varnames:
        assert np.allclose(grad[key], grad_num[key])


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

    # Compute the flux and its gradient using starry beta
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

    # Compute the flux and its gradient using starry beta
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


def test_ylm():
    # Define all arguments
    npts = 2
    ydeg = 1
    kwargs = {
        "y":        [0.1, 0.2, 0.3],
        "theta":    [0., 30.],
        "xo":       [0.15, 0.15],
        "yo":       [0., 0.1],
        "zo":       [1.0, 1.0],
        "ro":       [0.1, 0.1]
    }
    theano_kwargs = {}
    for key in kwargs.keys():
        theano_kwargs[key] = theano.shared(np.atleast_1d(kwargs[key]), name=key)
    
    # Compute the flux and its gradient using starry
    map_beta = starry_beta.Map(ydeg)
    map_beta[0, 0] = 1
    map_beta[1:, :] = kwargs.pop("y")
    kwargs.pop("zo")
    flux, grad = map_beta.flux(gradient=True, **kwargs)
    grad["y"] = grad["y"][1:, :]

    # Instantiate the theano op
    map = starry.Map(ydeg)
    model = map.flux(**theano_kwargs)

    # Compute the gradient using Theano
    varnames = sorted(theano_kwargs.keys())
    vars = [theano_kwargs[k] for k in varnames]
    computed = {}
    for var in varnames:
        deriv = [theano.function([], 
                 theano.grad(model[i], theano_kwargs[var]))() 
                 for i in range(npts)]
        if var == "y":
            computed[var] = np.transpose(deriv)
        else:
            computed[var] = np.sum(deriv, axis=0)

    # Compare
    for key in theano_kwargs.keys():
        if key == "zo":
            continue
        assert np.allclose(np.squeeze(grad[key]), np.squeeze(computed[key]))