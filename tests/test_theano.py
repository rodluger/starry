# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import theano
import theano.tensor as tt
from starry import DopplerMap


def test_doppler():
    # Define all arguments
    kwargs = {
        "y":        [0.25, 0.25, 0.25],
        "theta":    30.0,
        "inc":      75.0,
        "obl":      30.0,
        "alpha":    0.40,
        "veq":      3.00,
        "xo":       0.15,
        "yo":       0.2,
        "zo":       1.0,
        "ro":       0.1,
        "u":        [0.4, 0.26]
    }
    theano_kwargs = {}
    for key in kwargs.keys():
        theano_kwargs[key] = tt.as_tensor_variable(np.float64(kwargs[key]), name=key)

    # Compute the rv and its gradient using starry
    map = DopplerMap(ydeg=1, udeg=2) 
    map.inc = kwargs.pop("inc")
    map.obl = kwargs.pop("obl")
    map.alpha = kwargs.pop("alpha")
    map.veq = kwargs.pop("veq")
    map[1:, :] = kwargs.pop("y")
    map[1:] = kwargs.pop("u")
    rv, grad = map.rv(gradient=True, **kwargs)

    # Instantiate the theano op
    model = map.rv_op(**theano_kwargs)

    # Compare
    for key in theano_kwargs.keys():
        if key == "zo":
            continue
        assert np.allclose(
            np.squeeze(grad[key]),
            np.squeeze(theano.grad(model[0], theano_kwargs[key]).eval())
        ), key


def test_doppler_no_longer_broken():
    """
    If we give two variables the same value, theano *sums* over their
    gradients, and yields the *same value* for the gradient with respect
    to both one of them. Why is that??

    """
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
        theano_kwargs[key] = theano.shared(np.float64(kwargs[key]), name=key)

    # Compute the rv and its gradient using starry
    map = DopplerMap(ydeg=1, udeg=2)
    map.inc = kwargs.pop("inc")
    map.obl = kwargs.pop("obl")
    map.alpha = kwargs.pop("alpha")
    map.veq = kwargs.pop("veq")
    map[1:, :] = kwargs.pop("y")
    map[1:] = kwargs.pop("u")
    rv, grad = map.rv(gradient=True, **kwargs)
    print(grad["alpha"], grad["veq"])

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
