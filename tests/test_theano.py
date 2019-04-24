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
